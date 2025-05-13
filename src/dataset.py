# src/dataset.py
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import json
import cv2
from torchvision import transforms

class BiofilmDataset(Dataset):
    def __init__(self, image_dir, mask_dir, split_json_path, augmentation=False, mode="train"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augmentation = augmentation
        self.mode = mode

        with open(split_json_path) as f:
            self.split_json = json.load(f)
        
        if (mode == "train"):
            self.image_paths = self.split_json["train"]["images"]
            self.mask_paths = self.split_json["train"]["masks"]
            
        if (mode == "test"):
            self.image_paths = self.split_json["test"]["images"]
            self.mask_paths = self.split_json["test"]["masks"]
            
        self.transform = self.get_transforms()

    def get_transforms(self):
        aug_list = []
        if self.augmentation:
            aug_list.append(A.HorizontalFlip(p=0.5))
            aug_list.append(A.VerticalFlip(p=0.5))
        
        aug_list += [
            A.Normalize(mean=0, std=1, max_pixel_value = 255, normalization = "standard"), 
            # img = (img - mean * 255) / (std * 255)
            ToTensorV2()
        ]
        return A.Compose(aug_list)

    def make_clahe(self, img):
        img_np = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_np)
        return Image.fromarray(img_clahe)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')  # grayscale
        mask = Image.open(self.mask_paths[idx]).convert('L')
        image = self.make_clahe(image)
        
        mask = np.array(mask)
        mask = (mask > 127).astype(np.uint8)

        augmented = self.transform(image=np.array(image), mask=mask)
        image = augmented['image']
        mask = augmented['mask'].long()  # ??? для CrossEntropyLoss

        return image, mask

def visualize_sample(dataset, n=3):
    """Визуализация первых N примеров"""
    for i in range(min(n, len(dataset))):
        img, mask = dataset[i]
        
        if isinstance(img, torch.Tensor):
            img = img.numpy().squeeze()
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy().squeeze()
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title(f"Image {i}")
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title(f"Mask {i}")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        
        plt.show()