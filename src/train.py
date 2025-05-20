# src/train.py
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def trainModel(model, train_loader, device, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["learning_rate"])
    
    if (cfg["train"]["criterion"] == "CrossEntropyLoss"):
        criterion = torch.nn.CrossEntropyLoss() 
    else:
        criterion = torch.nn.MSELoss()
    
    epochs = cfg["train"]["epochs"]
    train_losses = []
    
    print(f"""---> TRAINING CONFIGURATION <---
          Loss functions is: {criterion}
          Batch size is: {cfg["train"]["batch_size"]}
          Device is: {device}
          Epoch count is: {epochs}
    """)
    
    for epoch in range(epochs):
        model.train()
        
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100):
            images, imgpaths, masks, maskpaths, color_masks, cmaskspaths = batch
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_losses.append(train_loss / len(train_loader)) 
        visualize_predictions(model, train_loader, cfg, device, mode="train", epoch=epoch)

        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Loss: {train_losses[-1]:.4f}")
        
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"{cfg['paths']['output_dir']}/model_epoch_{epoch}.pth")
            print(f"---> Model saved on epoch {epoch}")
        
    torch.save(model.state_dict(), f"{cfg['paths']['output_dir']}/final_model_epoch_{epochs}.pth")
    print(f"---> FINAL model saved on epoch {epochs} in {cfg['paths']['output_dir']}/final_model_epoch_{epochs}.pth")
    plot_loss_graph(train_losses, cfg)

def visualize_predictions(model, data_loader, cfg, device, mode="train", epoch=0, num_images=5):
    model.eval()

    images, imgpaths, masks, maskpaths, color_masks, cmaskspaths = next(iter(data_loader)) 
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
    plt.savefig(f"{cfg['paths']['output_dir']}/loss_curve.png")
    plt.close()
    print(f"---> Loss curve saved in {cfg['paths']['output_dir']}/loss_curve.png")

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
    plt.savefig(f"{cfg['paths']['output_dir']}/loss_curve.png")
    plt.close()



