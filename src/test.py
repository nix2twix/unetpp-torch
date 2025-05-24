from tempfile import tempdir
import torch
from pathlib import Path
import json
from torch.utils.data import DataLoader
from src.dataset import BiofilmDataset, TestDataset, splitDatasetInDirs, leaveOneSEMimageOut
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os
from PIL import Image
from src.model import build_model
from src.preprocessing import binarizeMaskDir, cropLineBelow, slidingWindowPatchDir, slidingWindowPatch
import re
pattern = r'\.(\d+)_(\d+)\.png$'

def load_checkpoint(model, checkpoint_path):
    model = nn.DataParallel(model) 
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.module    

    print(f"Model loaded from: {checkpoint_path}")

def iou_score(preds, masks, eps=1e-6): 
    intersection = torch.logical_and(preds == 1, masks == 1).sum(dim=(1, 2))
    union = torch.logical_or(preds == 1, masks == 1).sum(dim=(1, 2))
    iou = intersection / (union + eps)
    return iou

def dice_score(preds, masks, eps=1e-6):
    intersection = torch.logical_and(preds == 1, masks == 1).sum(dim=(1, 2))
    total = preds.sum(dim=(1, 2)) + masks.sum(dim=(1, 2))
    dice = (2 * intersection) / (total + eps)
    return dice

def fmeasure_score(preds, masks, positiveClass = 1, negativeClass = 0, eps=1e-6):
    TP = torch.logical_and(preds == positiveClass, masks == positiveClass).sum(dim=(1, 2))
    FP = torch.logical_and(preds == positiveClass, masks == negativeClass).sum(dim=(1, 2))
    TN = torch.logical_and(preds == negativeClass, masks == negativeClass).sum(dim=(1, 2))
    FN = torch.logical_and(preds == negativeClass, masks == positiveClass).sum(dim=(1, 2))
    Fmeasure = (2 * TP) / (2*TP + FP + FN + eps)
    return Fmeasure

def p4measure_score(preds, masks, eps=1e-6):
    TP = torch.logical_and(preds == 1, masks == 1).sum(dim=(1, 2))
    FP = torch.logical_and(preds == 1, masks == 0).sum(dim=(1, 2))
    TN = torch.logical_and(preds == 0, masks == 0).sum(dim=(1, 2))
    FN = torch.logical_and(preds == 0, masks == 1).sum(dim=(1, 2))
    invTP = 1.0 / (TP + eps)
    invTN = 1.0 / (TN + eps)
    p4measure = 4.0 / ((invTP + invTN) * (FP + FN) + 4)
    return p4measure

def test_model(model, test_loader, test_dataset, device, visualize=True, max_vis=5, 
               saveDir = None):
    model.eval()
    total_iou = 0.0
    total_p4 = 0.0
    count = 0
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0

    total_fmeasurePos1 = 0
    total_fmeasurePos0 = 0
    
    with torch.no_grad():
        for i, (images, imgpaths, masks, maskpaths, color_masks, cmaskspaths) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            TPinBatch = torch.logical_and(preds == 1, masks == 1).sum(dim=(1, 2))
            FPinBatch = torch.logical_and(preds == 1, masks == 0).sum(dim=(1, 2))
            TNinBatch = torch.logical_and(preds == 0, masks == 0).sum(dim=(1, 2))
            FNinBatch = torch.logical_and(preds == 0, masks == 1).sum(dim=(1, 2))
           
            iouInBatch = iou_score(preds, masks)
            p4InBatch = p4measure_score(preds, masks)

            fmeasureInBatchPos1 = fmeasure_score(preds, masks, 1, 0)
            fmeasureInBatchPos0 = fmeasure_score(preds, masks, 0, 1)
            
            for idx in range(images.size(0)):
                img_path = imgpaths[idx]
                img_name = os.path.basename(img_path)
                
                match = re.search(pattern, img_name)
                if match:
                    x = match.group(1)
                    y = match.group(2)

                orig_img = np.array(Image.open(img_path).convert("L"))
                color_img = np.array(color_masks[idx])
                binary_mask = masks[idx].cpu().numpy()
                pred_mask = preds[idx].cpu().numpy()
           
                
                total_iou += iouInBatch[idx]
                total_p4 += p4InBatch[idx]
                total_fmeasurePos1 += fmeasureInBatchPos1[idx]
                total_fmeasurePos0 += fmeasureInBatchPos0[idx]
                total_tp += TPinBatch[idx]
                total_fp += FPinBatch[idx]
                total_tn += TNinBatch[idx]
                total_fn += FNinBatch[idx]

                count += 1
                    
                print(f'---> {img_name} <---')
                print(f'IoU: {iouInBatch[idx]:.8f}, P4: {p4InBatch[idx]:.8f}')
                print(f'F1 (1): {fmeasureInBatchPos1[idx]:.8f}, F1 (0): {fmeasureInBatchPos0[idx]:.8f}')
                print(f'Contingency table (confusion matrix): \n' +
                      f'\t TP: {TPinBatch[idx]} \t FN: {FNinBatch[idx]}\n' + 
                      f'\t FP: {FPinBatch[idx]} \t TN {TNinBatch[idx]}')
                
                fig, axs = plt.subplots(1, 4, figsize=(18, 4))
                axs[0].imshow(orig_img, cmap="gray")
                axs[0].set_title(f"Orig: {img_name}")
                axs[1].imshow(binary_mask, cmap="gray")
                axs[1].set_title(f"GT Mask")
                axs[2].imshow(pred_mask, cmap="gray")
                axs[2].set_title(f"Pred Mask")
                axs[3].imshow(color_img)
                axs[3].set_title("Color Mask")

                for ax in axs:
                    ax.axis('off')
                plt.show()
                
    invTP = 1.0 / total_tp
    invTN = 1.0 / total_tn

    print(f"\n---> MEAN <---\nIoU: {total_iou / count:.10f}\nP4: {total_p4 / count:.10f}")
    print(f"F1 (1): {total_fmeasurePos1 / count:.10f} \nF1 (0): {total_fmeasurePos0 / count:.10f}")
    print(f"\n---> TOTAL \nIoU: {total_tp / (total_tp + total_fp + total_fn):.10f}\nP4: {4.0 / ((invTP + invTN) * (total_fp + total_fn) + 4):.10f}")
    print(f"F1 (1): {(2*total_tp) / (2*total_tp + total_fp + total_fn):.10f} \nF1 (0): {(2*total_tn) / (2*total_tn + total_fp + total_fn):.10f}")

