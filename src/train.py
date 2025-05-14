# src/train.py
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import calculate_iou, calculate_dice
import numpy as np

def train_model(model, train_loader, device, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["learning_rate"])
    
    if (cfg["train"]["critetrion"] == "CrossEntropyLoss"):
        criterion = torch.nn.CrossEntropyLoss() 
    else:
        criterion = torch.nn.MSELoss()
    
    epochs = cfg["train"]["epochs"]
    train_losses = []
    train_ious = []
    train_dices = []
    
    print(f"""---> TRAINING CONFIGURATION <---
          Loss functions is: {criterion}
          Batch size is:
          Device is:
          Learning rate is:
          Epoch counts is: {epochs}
    """)
    
    # Процесс обучения
    for epoch in range(epochs):
        model.train()
        
        train_loss = 0
        iou_total = 0
        dice_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100):
            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = outputs.argmax(dim=1) # softmax
            iou_total += calculate_iou(preds, masks)
            dice_total += calculate_dice(preds, masks)
            
        train_losses.append(train_loss / len(train_loader)) #зачем делить?
        train_ious.append(iou_total / len(train_loader))
        train_dices.append(dice_total / len(train_loader))

        visualize_predictions(model, train_loader, cfg, device, mode="train", epoch=epoch)

        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Loss: {train_losses[-1]:.4f}, "
              f"IoU: {train_ious[-1]:.4f}, "
              f"Dice: {train_dices[-1]:.4f}")
        
        torch.save(model.state_dict(), f"{cfg['paths']['output_dir']}/model_epoch_{epoch+1}.pth")

    plot_loss_graph(train_losses, cfg)

def visualize_predictions(model, data_loader, cfg, device, mode="train", epoch=0, num_images=5):
    model.eval()

    images, masks = next(iter(data_loader)) 
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(dim=1)  # Переводим в классы (0 или 1)

    for i in range(min(num_images, len(images))):
        image = images[i].cpu().numpy().transpose(1, 2, 0)
        mask = masks[i].cpu().numpy()
        pred = preds[i].cpu().numpy()

        fig, ax = plt.subplots(1, 3, figsize=(12, 5))
        ax[0].imshow(image.squeeze(), cmap="gray")
        ax[0].set_title("Image")
        ax[1].imshow(mask.squeeze(), cmap="gray")
        ax[1].set_title("Ground Truth")
        ax[2].imshow(pred.squeeze(), cmap="gray")
        ax[2].set_title("Prediction")
        plt.show() 
        plt.savefig(f"{cfg['paths']['output_dir']}/{mode}_epoch_{epoch+1}_image_{i+1}.png")
        plt.close()


def plot_loss_graph(train_losses, cfg):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Train Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.show() 
    # plt.savefig(f"{cfg['paths']['output_dir']}/loss_curve.png")
    plt.close()



