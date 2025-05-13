# src/utils.py
import torch
import numpy as np
import random
import json
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np

# --- IoU ---
def calculate_iou(preds, masks, eps=1e-7):
    intersection = torch.logical_and(preds == 1, masks == 1).sum(dim=(1, 2))
    union = torch.logical_or(preds == 1, masks == 1).sum(dim=(1, 2))
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

# --- Dice ---
def calculate_dice(preds, masks, eps=1e-7):
    intersection = (preds * masks).sum(dim=(1, 2))
    total = preds.sum(dim=(1, 2)) + masks.sum(dim=(1, 2))
    dice = (2. * intersection + eps) / (total + eps)
    return dice.mean().item()

