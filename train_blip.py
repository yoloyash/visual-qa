import torch
from tqdm import tqdm
import argparse
from transformers import BlipProcessor, BlipForQuestionAnswering,BlipImageProcessor
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR
from loguru import logger 
import time 
import os
import numpy as np 
import matplotlib.pyplot as plt
from datetime import datetime
from dataset import VQADatasetBLIP, collate_fn_blip



def train(model, dataloader, optimizer, scheduler, device):

    total_loss = 0.0
    model.train()
    
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    train_loss = total_loss / len(dataloader.dataset)
    return train_loss


def val(model, dataloader, device):
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

    val_loss = total_loss / len(dataloader.dataset)
    return val_loss


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_name", type=str, default="blip-vqa")
    parser.add_argument("--train_json_path", type=str, help="Path to the training data")
    parser.add_argument("--val_json_path", type=str, help="Path to the validation data")
    parser.add_argument("--model_save_dir", type=str, default="/home/ec2-user/vqa_project/runs", help="Path to save the model")

    args = parser.parse_args()

    folder_name = datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + args.model_name
    run_dir = os.path.join(args.model_save_dir, folder_name)
    check_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)

    text_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    image_processor = BlipImageProcessor.from_pretrained("Salesforce/blip-vqa-base")

    train_dataset = VQADatasetBLIP(args.train_json_path, text_processor, image_processor)
    val_dataset = VQADatasetBLIP(args.val_json_path, text_processor, image_processor)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_blip)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_blip)

    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

    device = torch.device(args.device)
    model.to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)
    image_mean, image_std = image_processor.image_mean, image_processor.image_std


    best_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"Epoch [{epoch + 1}/{args.epochs}]:")
        start_time = time.time()
        
        train_loss = train(model, train_loader, optimizer, scheduler, device)
        val_loss = val(model, val_loader, device)

        epoch_time = time.time() - start_time
        avg_time_step = epoch_time / (len(train_loader) + len(val_loader))

        if val_loss < best_loss:
            logger.info(f"val_acc improved from {best_loss:.5f} to {val_loss:.5f}")
            best_loss = val_loss
            checkpoint_name = f"{args.model_name}_{best_loss:.4f}.pth"
            state_dict = {
                "state_dict" : model.state_dict(), 
                "optimizer" : optimizer.state_dict(),
                "img_mean" : image_mean,
                "img_std" : image_std
            }
            torch.save(state_dict, os.path.join(check_dir, checkpoint_name))
            logger.info(f"Model Saved as {checkpoint_name}")