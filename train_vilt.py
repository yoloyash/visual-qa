import os
from datetime import datetime
import torch
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import ViltProcessor, ViltForQuestionAnswering
from torch.utils.data import DataLoader
import time
from loguru import logger
import matplotlib.pyplot as plt

from utils import get_class_mapping_vilt, add_label_score_vilt
from dataset import VizWizDatasetViLT
from model import get_optimizer_and_scheduler


def train(dataloader, model, optimizer, scheduler, device):
    total_samples = len(dataloader.dataset)
    # Define loss
    total_loss = 0
    total_acc = 0
    model.train()
    for batch in dataloader:
        # get the inputs;
        inputs = {k:v.to(device) for k,v in batch.items()}
        # forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        # backward and optimize
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.logits.argmax(-1)
        scores, labels = batch["labels"].to(device).topk(10, -1)
        
        # Calculate accuracy
        for idx in range(len(scores)):
            total_acc += min(scores[idx][preds[idx] == labels[idx]].sum(),1)
        #scheduler.step()
        
    train_loss = total_loss / total_samples
    train_acc = total_acc / total_samples

    return train_acc ,train_loss


def val(dataloader, model, device):
    total_samples = len(dataloader.dataset)
    # Define loss and accuracy
    total_loss = 0
    total_acc = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k:v.to(device) for k,v in batch.items()}
            outputs = model(**inputs)

            # Calculate loss
            loss = outputs.loss
            total_loss += loss.item()

            # Get top predict for each question
            preds = outputs.logits.argmax(-1)
            # Get ground truth answers for each questiojn
            scores, labels = batch["labels"].to(device).topk(10, -1)
            # Calculate accuracy
            for idx in range(len(scores)):
                total_acc += min(scores[idx][preds[idx] == labels[idx]].sum(),1)
            
    val_acc = total_acc / total_samples
    val_loss = total_loss / total_samples

    return val_acc, val_loss


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Finetune ViLT model for VQA")

    parser.add_argument("--train_json_path", type=str, help="Path to the training data")
    parser.add_argument("--val_json_path", type=str, help="Path to the validation data")
    parser.add_argument("--class_mapping_file_path", type=str, help="Path to the class mapping file")
    parser.add_argument("--model_save_dir", type=str, default="/home/ec2-user/vqa_project/runs", help="Path to save the model")

    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--base_lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--lr_patience", type=int, default=3, help="Patience for learning rate scheduler")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for optimizer")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer for training")

    parser.add_argument("--model_name", type=str, default="vilt", help="Name of the model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")

    args = parser.parse_args()

    folder_name = datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + args.model_name
    run_dir = os.path.join(args.model_save_dir, folder_name)
    check_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)

    label2id, id2label = get_class_mapping_vilt(args.class_mapping_file_path)
    # print(label2id)

    train_data = pd.read_json(args.train_json_path)
    val_data = pd.read_json(args.val_json_path)

    train_data = add_label_score_vilt(train_data, label2id)
    val_data = add_label_score_vilt(val_data, label2id)

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm",
                                                id2label=id2label,
                                                label2id=label2id)
    
    train_dataset = VizWizDatasetViLT(train_data, processor, label2id)
    val_dataset = VizWizDatasetViLT(val_data, processor, label2id)

    train_dataloader = DataLoader(train_dataset, 
                                  collate_fn=lambda x : train_dataset.collate_fn(x), 
                                  batch_size=args.batch_size, 
                                  shuffle=True, 
                                  num_workers=2, 
                                  pin_memory=True, 
                                  drop_last=True)
    val_dataloader = DataLoader(val_dataset, 
                                collate_fn=lambda x : val_dataset.collate_fn(x), 
                                batch_size=args.batch_size,
                                shuffle=False, 
                                num_workers=2, 
                                pin_memory=True)
    
    
    device = torch.device(args.device)
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm",
                                                    id2label=id2label,
                                                    label2id=label2id)
    model.to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model, optim=args.optimizer, lr=args.lr)


    best_val_acc = 0

    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []

    for epoch in tqdm(range(args.epochs), desc="Training", total=args.epochs):
        
        print(f"Epoch [{epoch + 1}/{args.epochs}]:")

        start_time = time.time()
        train_acc, train_loss = train(train_dataloader, model, optimizer, scheduler, device)
        val_acc, val_loss = val(val_dataloader, model, device)

        epoch_time = time.time() - start_time
        # avg_step_time = epoch_time / (len(train_dataloader) + len(val_dataloader))

        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)

        logger.info(f"Train loss: {train_loss:.5f} - Val loss: {val_loss:.5f}")
        logger.info(f"Train Accuracy : {train_acc:.5f} - Val accuracy: {val_acc:.5f}\n")

        if val_acc > best_val_acc:
            logger.info(f"val_acc improved from {best_val_acc:.5f} to {val_acc:.5f}")
            best_val_acc = val_acc
            checkpoint_name = f"{args.model_name}_{best_val_acc:.4f}.pth"
            torch.save(model, os.path.join(check_dir, checkpoint_name))
            logger.info(f"Model Saved as {checkpoint_name}")
        
        scheduler.step(val_loss)

    # plot the training and validation loss and accuracies
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='train_acc')
    plt.plot(val_accuracies, label='val_acc')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig(os.path.join(run_dir, 'train_val_metrics.png'))

    logger.info("Training Completed!")
