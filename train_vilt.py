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
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils import get_class_mapping_vilt, add_label_score_vilt
from dataset import VizWizDatasetViLT, collate_fn_vilt
from model import get_optimizer_and_scheduler


def train(dataloader, model, optimizer, scheduler, device):
    total_samples = len(dataloader.dataset)
    total_loss = 0
    total_acc = 0
    corect_predictions = 0
    model.train()
    for batch in dataloader:
        inputs = {k:v.to(device) for k,v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.logits.argmax(-1)
        scores, labels = batch["labels"].to(device).topk(10, -1)
        # score shape : (batch_size, 10)
        # preds shape : (batch_size)
        # labels shape : (batch_size, 10)
        # total_acc += scores[preds==labels].sum() / len(preds)
        # Calculate accuracy
        for idx in range(len(scores)):
            total_acc += min(scores[idx][preds[idx] == labels[idx]].sum(),1)

        corect_predictions += (preds == labels[:, 0]).sum().item()
        #scheduler.step()
        
    train_loss = total_loss / total_samples
    train_acc = total_acc / total_samples
    normal_acc = corect_predictions / total_samples

    return train_acc ,train_loss, normal_acc


def val(dataloader, model, device):
    total_samples = len(dataloader.dataset)
    # Define loss and accuracy
    total_loss = 0
    total_acc = 0
    correct_predictions = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
        # for batch in tqdm(dataloader, desc="Validation"):
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
            correct_predictions += (preds == labels[:, 0]).sum().item()

    val_acc = total_acc / total_samples
    val_loss = total_loss / total_samples
    normal_acc = correct_predictions / total_samples

    return val_acc, val_loss, normal_acc


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

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    
    train_dataset = VizWizDatasetViLT(train_data, processor, label2id)
    val_dataset = VizWizDatasetViLT(val_data, processor, label2id)

    train_dataloader = DataLoader(train_dataset, 
                                #   collate_fn=lambda x : train_dataset.collate_fn(x), 
                                    collate_fn=lambda x : collate_fn_vilt(x, processor),
                                  batch_size=args.batch_size, 
                                  shuffle=True, 
                                  num_workers=2, 
                                  pin_memory=True, 
                                  drop_last=True)
    val_dataloader = DataLoader(val_dataset, 
                                # collate_fn=lambda x : val_dataset.collate_fn(x), 
                                collate_fn=lambda x : collate_fn_vilt(x, processor),
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
    train_normal_accuracies = []
    val_normal_accuracies = []

    for epoch in tqdm(range(args.epochs), desc="Training", total=args.epochs):
    # for epoch in range(args.epochs):
        
        print(f"Epoch [{epoch + 1}/{args.epochs}]:")

        start_time = time.time()
        train_acc, train_loss, train_normal_acc = train(train_dataloader, model, optimizer, scheduler, device)
        val_acc, val_loss, val_normal_acc = val(val_dataloader, model, device)

        epoch_time = time.time() - start_time
        # avg_step_time = epoch_time / (len(train_dataloader) + len(val_dataloader))

        # tensor to cpu to numpy
        # print(type(train_acc), type(train_loss), type(val_acc), type(val_loss))
        train_accuracies.append(train_acc.cpu().numpy())
        train_losses.append(np.array(train_loss))
        val_accuracies.append(val_acc.cpu().numpy())
        val_losses.append(np.array(val_loss))

        train_normal_accuracies.append(np.array(train_normal_acc))
        val_normal_accuracies.append(np.array(val_normal_acc))

        # if epoch==0:
        #     print(type(train_accuracies), type(train_losses), type(val_accuracies), type(val_losses))

        logger.info(f"Train loss: {train_loss:.5f} - Val loss: {val_loss:.5f}")
        logger.info(f"Train VizWiz Accuracy: {train_acc:.5f} - Val VizWiz accuracy: {val_acc:.5f}")
        logger.info(f"Train Normal Accuracy: {train_normal_acc:.5f} - Val Normal Accuracy: {val_normal_acc:.5f}\n")

        if val_acc > best_val_acc:
            logger.info(f"val_acc improved from {best_val_acc:.5f} to {val_acc:.5f}")
            best_val_acc = val_acc
            checkpoint_name = f"{args.model_name}_{best_val_acc:.4f}.pth"
            torch.save(model, os.path.join(check_dir, checkpoint_name))
            logger.info(f"Model Saved as {checkpoint_name}")
        
        scheduler.step(val_loss)

    # plot the training and validation loss and accuracies
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(train_losses, label='train_loss')
    # plt.plot(val_losses, label='val_loss')
    # plt.legend()
    # plt.title('Loss')
    # plt.subplot(1, 2, 2)
    # plt.plot(train_accuracies, label='train_acc')
    # plt.plot(val_accuracies, label='val_acc')
    # plt.legend()
    # plt.title('Accuracy')
    # plt.savefig(os.path.join(run_dir, 'train_val_metrics.png'))

    # plot both accuracies and normal accuracies and losses
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='train_acc')
    plt.plot(val_accuracies, label='val_acc')
    plt.plot(train_normal_accuracies, label='train_normal_acc')
    plt.plot(val_normal_accuracies, label='val_normal_acc')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig(os.path.join(run_dir, 'train_val_metrics.png'))



    logger.info("Training Completed!")
