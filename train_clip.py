import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dataset import VizWizDataset
from loguru import logger
from sklearn.preprocessing import OrdinalEncoder
import pickle
from model import get_optimizer_and_scheduler
from model import VQAModelV3Attn as VQAModel
import time
from itertools import combinations
import matplotlib.pyplot as plt


def accuracy_vqa(df, index, value):
    if value == None:
        return 0
    ans_list = [elem['answer'] for elem in df.iloc[index]['answers']]
    return np.divide(np.sum(np.minimum(np.count_nonzero(np.array(list(combinations(ans_list, 9))) == value, axis=1), 1)), 10)


def train(model, dataloader, criterion, optimizer, device, encoder):
    
    model.train()
    train_loss = 0
    accuracy = 0

    for index, x, answers, _ in dataloader:
        x = x.to(device) 
        answers = torch.as_tensor(encoder.transform(np.array(answers).reshape(-1, 1)).astype(int)).to(device).squeeze(1)
        
        # Forward Pass
        outputs = model(x).squeeze(1)
        loss = criterion(outputs, answers)
        
        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        
        # Update Weights
        optimizer.step()

        # Loss and Accuracy Calculations
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        predicted = encoder.inverse_transform(np.array(predicted.to('cpu')).reshape(-1,1))
        for ip, idx in enumerate(index):
            accuracy += accuracy_vqa(dataloader.dataset.df, int(idx), predicted[ip])

    train_loss /= len(dataloader.dataset)
    accuracy /= len(dataloader.dataset)
    
    return train_loss, accuracy


def validate(model, dataloader, criterion, device, encoder):

    model.eval()
    val_loss = 0
    accuracy = 0
    with torch.no_grad():
        for index, x, answers, _ in dataloader:
            x = x.to(device)
            answers = torch.as_tensor(encoder.transform(np.array(answers).reshape(-1, 1)).astype(int)).to(device).squeeze(1)

            # Forward Pass
            outputs = model(x).squeeze(1)
            loss = criterion(outputs, answers)
            
            # Loss and Accuracy Calculations
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            predicted = encoder.inverse_transform(np.array(predicted.to('cpu')).reshape(-1,1))
            for ip, idx in enumerate(index):
                accuracy += accuracy_vqa(dataloader.dataset.df, int(idx), predicted[ip])

    val_loss /= len(dataloader.dataset)
    accuracy /= len(dataloader.dataset)

    return val_loss, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CLIP-based VQA model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--input_dim", type=int, default=2048, help="Input dimension for the model")
    parser.add_argument("--hidden_dim", type=int, default=2048, help="Hidden dimension for the model")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer for training")
    parser.add_argument("--model_name", type=str, default="clip-rn50-1", help="Name of the model file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on")
    # data arguments
    parser.add_argument("--train_json_path", type=str, help="Path to the training data")
    parser.add_argument("--val_json_path", type=str, help="Path to the validation data")
    parser.add_argument("--model_save_dir", type=str, default="/home/ec2-user/vqa_project/runs", help="Path to save the model")

    args = parser.parse_args()

    # writer = SummaryWriter()
    folder_name = datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + args.model_name
    run_dir = os.path.join(args.model_save_dir, folder_name)
    check_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)

    train_dataset = VizWizDataset(args.train_json_path, filter_answerable=False)
    val_dataset = VizWizDataset(args.val_json_path, filter_answerable=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    logger.info("Data loaded Successfully!")

    ANSWER_CANDIDATES = train_dataset.df['final_answer'].nunique()
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=ANSWER_CANDIDATES)
    encoder.fit(np.array(train_dataset.df['final_answer']).reshape(-1, 1))

    # save the encoder for inference
    with open(os.path.join(run_dir, 'encoder.pkl'), 'wb') as f:
        pickle.dump(encoder, f)

    # output_dim will be the number of answer candidates + 1 for the unanswerable class
    output_dim = ANSWER_CANDIDATES + 1
    device = torch.device(args.device)

    model = VQAModel(args.input_dim, args.hidden_dim, output_dim).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer, lr_scheduler = get_optimizer_and_scheduler(model, args.optimizer, args.lr)

    best_val_acc = 0.0
    # patience = 10

    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []

    for epoch in tqdm(range(args.epochs), desc="Training", total=args.epochs):

        print(f"Epoch [{epoch + 1}/{args.epochs}]:")
        
        start_time = time.time()
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, encoder)
        val_loss, val_acc = validate(model, val_loader, criterion, device, encoder)

        epoch_time = time.time() - start_time
        avg_step_time = epoch_time / (len(train_loader) + len(val_loader))
    
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)

        if val_acc >  best_val_acc:
            logger.info(f"val_acc improved from {best_val_acc:.5f} to {val_acc:.5f}")
            best_val_acc = val_acc
            checkpoint_name = f"{args.model_name}_{best_val_acc:.4f}.pth"
            torch.save(model, os.path.join(check_dir, checkpoint_name))
            logger.info(f"Model Saved as {checkpoint_name}")
            #counter = 0
            # Save the model checkpoint)
        # else:
        #     counter += 1
        #     if counter >= patience:
        #         print(f"val_loss hasn't improved for {patience} epochs. Early stopping.")
        #         break

        logger.info(f"{int(np.round(epoch_time))}s {avg_step_time*1e3:.4f}ms/step - loss: {train_loss:.4f} - accuracy: {train_acc*100:.4f}% - val_loss: {val_loss:.4f} - val_accuracy: {val_acc*100:.4f}% - lr: {optimizer.param_groups[0]['lr']}")
        
        lr_scheduler.step(val_loss)
        print()

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