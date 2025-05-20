# src/dataset.py
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from os.path import isfile, join
from os import listdir
import json
import cv2
from torchvision import transforms
import shutil
import glob

class BiofilmDataset(Dataset):
    def __init__(self, image_dir, mask_dir, colored_mask_dir, augmentation=False, mode="train"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augmentation = augmentation
        self.mode = mode

        image_paths = [os.path.join(image_dir, f) for f in listdir(image_dir) if isfile(join(image_dir, f))]
        print(f"---> IMAGES LOAD: {len(image_paths)} from {image_dir}")

        mask_paths = [os.path.join(mask_dir, f) for f in listdir(mask_dir) if isfile(join(mask_dir, f))]
        print(f"---> MASKS LOAD: {len(mask_paths)} from {mask_dir}")

        colored_mask_paths = [os.path.join(colored_mask_dir, f) for f in listdir(colored_mask_dir) if isfile(join(colored_mask_dir, f))]
        print(f"---> COLORED MASKS LOAD: {len(colored_mask_paths)} from {colored_mask_dir}")

        self.image_paths = sorted(image_paths)
        self.mask_paths = sorted(mask_paths)
        self.colored_mask_paths = sorted(colored_mask_paths)
        self.transform = self.get_transforms()

    def get_transforms(self):
        aug_list = []
        if self.augmentation:
            aug_list.append(A.HorizontalFlip(p=0.5))
            aug_list.append(A.VerticalFlip(p=0.5))

        aug_list += [
            A.Normalize(mean = 0, std = 1), #image mode
            # img = (img - mean * 255) / (std * 255)
            ToTensorV2()
        ]
        return A.Compose(aug_list)

    def make_clahe(self, img):
        img_np = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_np)
        return Image.fromarray(img_clahe)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')  # grayscale
        mask = Image.open(self.mask_paths[idx]).convert('L')
        color_mask = Image.open(self.colored_mask_paths[idx]).convert("RGB")
        color_mask = np.array(color_mask)  # [H, W, 3]

        image = self.make_clahe(image)
        mask = np.array(mask)
        mask = (mask > 127).astype(np.uint8)

        augmented = self.transform(image=np.array(image), mask=mask)
        image = augmented['image']
        mask = augmented['mask'].long()  

        return image, self.image_paths[idx], mask, self.mask_paths[idx], color_mask, self.colored_mask_paths[idx]


def splitDatasetInDirs(trainSamplesCounts=80, testSamplesCounts=20,
                       sourceImgDir=None, sourceMasksDir=None, sourceColoredMasks = None, outputBaseDir=None):
    
    image_files = sorted(os.listdir(sourceImgDir))
    total = len(image_files)
    train_count = int(total * trainSamplesCounts / 100)

    train_files = image_files[:train_count]
    test_files = image_files[train_count:]

    train_img_dir = os.path.join(outputBaseDir, f"train_{trainSamplesCounts}", "images")
    train_mask_dir = os.path.join(outputBaseDir, f"train_{trainSamplesCounts}", "masks")
    train_colored_mask_dir = os.path.join(train_mask_dir, "colored-masks")
    test_img_dir = os.path.join(outputBaseDir, f"test_{testSamplesCounts}", "images")
    test_mask_dir = os.path.join(outputBaseDir, f"test_{testSamplesCounts}", "masks")
    test_colored_mask_dir = os.path.join(test_mask_dir, "colored-masks")
  
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(train_colored_mask_dir, exist_ok=True)

    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_mask_dir, exist_ok=True)
    os.makedirs(test_colored_mask_dir, exist_ok=True)

    for file in train_files:
        if not os.path.basename(file).startswith('.'):
            shutil.copy(os.path.join(sourceImgDir, file), os.path.join(train_img_dir, file))
            shutil.copy(os.path.join(sourceMasksDir, file), os.path.join(train_mask_dir, file))
            shutil.copy(os.path.join(sourceColoredMasks, file), os.path.join(train_colored_mask_dir, file))
    for file in test_files:
        if not os.path.basename(file).startswith('.'):
            shutil.copy(os.path.join(sourceImgDir, file), os.path.join(test_img_dir, file))
            shutil.copy(os.path.join(sourceMasksDir, file), os.path.join(test_mask_dir, file))
            shutil.copy(os.path.join(sourceColoredMasks, file), os.path.join(test_colored_mask_dir, file))

    print(f"Всего файлов: {total}")
    print(f"Train: {len(train_files)} → {train_img_dir}")
    print(f"Test: {len(test_files)} → {test_img_dir}")

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