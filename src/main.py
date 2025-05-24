# main.py
import json
from logging import config
from PIL import Image
import torch
import os
from numpy import save
from torch.utils.data import DataLoader
from src.dataset import BiofilmDataset, TestDataset, splitDatasetInDirs, leaveOneSEMimageOut
from src.model import build_model
from src.train import trainModel
from src.preprocessing import binarizeMaskDir, cropLineBelow, slidingWindowPatchDir, slidingWindowPatch
from src.test import test_model, load_checkpoint
import matplotlib.pyplot as plt
import re
import cv2
pattern = r'\.(\d+)_(\d+)\.png$'
import numpy as np

def main():
    # ---> TEST ONE SEM-IMAGE
    basePath = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project"
    
    imagePath = basePath + r"\dataset\testSourceDir\images"
    maskPath = basePath + r"\dataset\testSourceDir\masks-binarized"
    coloredMaskPath = basePath + r"\dataset\testSourceDir\masks"
    configPath = basePath + r"\unet-project-torch\config\test_config.json"
    checkpointPath = basePath + r"\experiments\leaveOneBSEout\1-BSE-1k-T1\final_model_epoch_300_1BSE.pth"
    
    testOneSEMimage(imgDir = imagePath,
                       maskDir = maskPath,
                       coloredMaskDir = coloredMaskPath,
                       outputDir = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project",
                       configPath = configPath,
                       checkpointPath = checkpointPath
    )
 
def testOneSEMimage(imgDir = None, maskDir = None, coloredMaskDir = None, outputDir = None, 
                       configPath = None, checkpointPath= None, targetRGB=(36, 179, 83)):

    imgProcessedDir = slidingWindowPatchDir(imgDir = imgDir, 
                       imgMode= "L",
                       patch_size = (512, 512),
                       stride = (128, 128),
                       save_dir= r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\process\images",
                       visualize=False
    )
    maskProcessedDir = slidingWindowPatchDir(imgDir = maskDir, 
                       imgMode= "L",
                       patch_size = (512, 512),
                       stride = (128, 128),
                       save_dir= r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\process\masks",
                       visualize=False
    )
    coloredMasksProcessedDir = slidingWindowPatchDir(imgDir = coloredMaskDir, 
                       imgMode= "RGB",
                       patch_size = (512, 512),
                       stride = (128, 128),
                       save_dir= r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\process\colored-masks",
                       visualize=False
    )

    with open(configPath) as f:
        cfg = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_dataset = BiofilmDataset(
        image_dir=imgProcessedDir,
        mask_dir=maskProcessedDir,
        colored_mask_dir=coloredMasksProcessedDir,
        mode="test"
    )
    test_loader = DataLoader(test_dataset,
                             batch_size=2,
                             shuffle=False)

    model = build_model(cfg["model"]).to(device)   
    load_checkpoint(model, checkpointPath)
    model.eval()
    
    height, width = 1792, 2560 
    probsCount = np.zeros((height, width), dtype=float)
    probsValues = np.zeros((height, width), dtype=float)
    predsValues = np.zeros((height, width), dtype=float)

    imgName = os.path.splitext(os.path.basename(os.listdir(imgDir)[0]))[0]
    imgPath =os.path.join(imgDir, os.listdir(imgDir)[0])
    maskPath = os.path.join(maskDir, os.listdir(maskDir)[0])
    coloredMaskPath= os.path.join(coloredMaskDir, os.listdir(coloredMaskDir)[0])
    
    with torch.no_grad():
        for i, (images, imgpaths, masks, maskpaths, color_masks, cmaskspaths) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            outputs = outputs.cpu()
            
            for idx in range(images.size(0)):
                img_path = imgpaths[idx]
                img_name = os.path.basename(img_path)
                
                match = re.search(pattern, img_name)
                if match:
                    x = int(match.group(1))
                    y = int(match.group(2))
                
                output_np = outputs[idx].numpy()[1]
                
                probsValues[y:y+512, x:x+512] += output_np
                probsCount[y:y+512, x:x+512] += 1
                print(f'---> {img_name} <---')

    
    probsValues = probsValues / probsCount # вероятности
    predsValues = (probsValues > 0.5).astype(np.uint8) 
   
    mask = Image.open(maskPath).convert("L")
    mask = cropLineBelow(mask, countPx=128)
    maskValues = (np.array(mask) > 127).astype(np.uint8)
    
    print("Уникальные значения predsValues:", np.unique(predsValues))
    print("Уникальные значения maskValues:", np.unique(maskValues))
    
    TP = np.logical_and(predsValues == 1, maskValues == 1).sum()
    FP = np.logical_and(predsValues == 1, maskValues == 0).sum()
    TN = np.logical_and(predsValues == 0, maskValues == 0).sum()
    FN = np.logical_and(predsValues == 0, maskValues == 1).sum()

    # METRICS
    print(f'---> {imgName} <---')
    print(f'Contingency table (confusion matrix): \n' +
            f'\t TP: {TP} \t FN: {FN}\n' + 
            f'\t FP: {FP} \t TN {TN}')
    invTP = 1 / (TP + 1e-7)
    invTN = 1 / (TN + 1e-7)
    print(f"IoU: {TP / (TP + FP + FN):.10f}\tP4: {4.0 / ((invTP + invTN) * (FP + FN) + 4):.10f}")
    print(f"F1 (1): {(2*TP) / (2*TP + FP + FN):.10f} \tF1 (0): {(2*TN) / (2*TN + FP + FN):.10f}")


    # VIZUALIZATION
    origImg = Image.open(imgPath).convert("L")
    origImg = cropLineBelow(origImg, countPx=128)
    coloredMask = Image.open(coloredMaskPath).convert("RGB")
    coloredMask = cropLineBelow(coloredMask, countPx=128)
    
    predMask = (probsValues * 255).astype(np.uint8)            
    binMask = (predsValues * 255).astype(np.uint8)
    
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    axs[0].imshow(origImg, cmap='gray')
    axs[0].set_title(f"Original {imgName}")
    axs[1].imshow(predMask, cmap='gray')
    axs[1].set_title("Predicted Mask")
    axs[2].imshow(binMask, cmap='gray', vmin=0, vmax=255)
    axs[2].set_title("Predicted Binary Mask")
    axs[3].imshow(mask, cmap='gray')
    axs[3].set_title("Original GT-Mask")
    axs[4].imshow(coloredMask)
    axs[4].set_title("Original Colored Mask")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    mask_color = (36/255, 179/255, 83/255, 0.5) 
    gt_colored = np.zeros((*maskValues.shape, 4))
    gt_colored[maskValues == 1] = mask_color
    pred_colored = np.zeros((*binMask.shape, 4))
    pred_colored[binMask == 255] = mask_color

    fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    axs[0,0].imshow(origImg, cmap='gray')
    axs[0,0].imshow(gt_colored)
    axs[0,0].set_title("Original with GT Mask")
    axs[0,0].axis('off')

    axs[0,1].imshow(origImg, cmap='gray')
    axs[0,1].imshow(pred_colored)
    axs[0,1].set_title("Original with Predicted Mask")
    axs[0,1].axis('off')

    axs[1,0].imshow(mask, cmap='gray')
    axs[1,0].set_title("GT Mask")
    axs[1,0].axis('off')

    axs[1,1].imshow(binMask, cmap='gray', vmin=0, vmax=255)
    axs[1,1].set_title("Predicted Binary Mask")
    axs[1,1].axis('off')

    plt.tight_layout()
    plt.show()

    if (outputDir != None):
        predMaskImage = Image.fromarray(predMask, mode='L')
        predMaskImage.save(f"{outputDir}/{imgName}-processed.png")
                    
        binMaskImage = Image.fromarray(binMask, mode='L')
        binMaskImage.save(f"{outputDir}/{imgName}-result.png")

def mainGlobal():
    
    # ---> LEAVE ONE SEM-IMAGE OUT
    """
    Take one SEM-image for test before processing.
    """
    leaveOneSEMimageOutPath = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\source\images\BSE\9-BSE-1k-T1.bmp"
    oneSEMimageMaskPath = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\source\masks\BSE\9-BSE-1k-T1.png"
    datasetPath = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset"
    leaveOneSEMimageOut(leaveOneSEMimageOutPath, oneSEMimageMaskPath, datasetPath)

    # ---> MASKS PREPROCESSING
    """ 
    Binarization colored masks
    for target class
    """
    masksList = {(0, 0, 0): 'Background (black)',
                 (255, 0, 0): 'Defect (red)',
                 (184, 61, 245): 'Single bacteria (purple)',
                 (36, 179, 83): 'Biofilm (green)',
                 (221, 255, 51): 'Intermediate stage (yellow)'
    }
    
    # FOR TRAIN DATASET
    masksDirPath = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\source\masks\BSE"
    
    binarizeMaskDir(
    masksDirPath,
    "RGB",
    masksList,
    targetClassColor=(36, 179, 83),
    secondClassColor=(0, 0, 0),
    saveDir=r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\masks\uncroped",
    vizN=1
    )
    
    # FOR TEST DATASET  
    masksDirPath = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\testSourceDir\masks"

    binarizeMaskDir(
    masksDirPath,
    "RGB",
    masksList,
    targetClassColor=(36, 179, 83),
    secondClassColor=(0, 0, 0),
    saveDir=r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\testSourceDir\masks-binarized",
    vizN=1
    )

    # ---> CROPPING TRAIN IMAGES AND MASKS
    """ 
    Croping black line below,
    cutting img and masks with sliding window
    """
    imagesDirPath = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\source\images\BSE"
    slidingWindowPatchDir(imgDir = imagesDirPath, 
                          imgMode='L',
                          patch_size = (512, 512),
                          stride = (128, 128),
                          save_dir=r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\images",
                          visualize=False
    )

    coloredMasksDirPath = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\source\masks\BSE"
    slidingWindowPatchDir(imgDir = coloredMasksDirPath, 
                          imgMode='RGB',
                          patch_size = (512, 512),
                          stride = (128, 128),
                          save_dir=r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\masks\colored-masks",
                          visualize=False
    )

    masksDirPath = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\masks\uncroped"
    slidingWindowPatchDir(imgDir = masksDirPath, 
                          imgMode='L',
                          patch_size = (512, 512),
                          stride = (128, 128),
                          save_dir=r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\masks",
                          visualize=False
    )# ---> LEAVE ONE SEM-IMAGE OUT
    """
    Take one SEM-image for test before processing.
    """
    leaveOneSEMimageOutPath = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\source\images\BSE\9-BSE-1k-T1.bmp"
    oneSEMimageMaskPath = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\source\masks\BSE\9-BSE-1k-T1.png"
    datasetPath = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset"
    leaveOneSEMimageOut(leaveOneSEMimageOutPath, oneSEMimageMaskPath, datasetPath)

    # ---> MASKS PREPROCESSING
    """ 
    Binarization colored masks
    for target class
    """
    masksList = {(0, 0, 0): 'Background (black)',
                 (255, 0, 0): 'Defect (red)',
                 (184, 61, 245): 'Single bacteria (purple)',
                 (36, 179, 83): 'Biofilm (green)',
                 (221, 255, 51): 'Intermediate stage (yellow)'
    }
    
    # FOR TRAIN DATASET
    masksDirPath = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\source\masks\BSE"
    
    binarizeMaskDir(
    masksDirPath,
    "RGB",
    masksList,
    targetClassColor=(36, 179, 83),
    secondClassColor=(0, 0, 0),
    saveDir=r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\masks\uncroped",
    vizN=1
    )
    
    # FOR TEST DATASET  
    masksDirPath = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\testSourceDir\masks"

    binarizeMaskDir(
    masksDirPath,
    "RGB",
    masksList,
    targetClassColor=(36, 179, 83),
    secondClassColor=(0, 0, 0),
    saveDir=r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\testSourceDir\masks-binarized",
    vizN=1
    )

    # ---> CROPPING TRAIN IMAGES AND MASKS
    """ 
    Croping black line below,
    cutting img and masks with sliding window
    """
    imagesDirPath = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\source\images\BSE"
    slidingWindowPatchDir(imgDir = imagesDirPath, 
                          imgMode='L',
                          patch_size = (512, 512),
                          stride = (128, 128),
                          save_dir=r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\images",
                          visualize=False
    )

    coloredMasksDirPath = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\source\masks\BSE"
    slidingWindowPatchDir(imgDir = coloredMasksDirPath, 
                          imgMode='RGB',
                          patch_size = (512, 512),
                          stride = (128, 128),
                          save_dir=r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\masks\colored-masks",
                          visualize=False
    )

    masksDirPath = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\masks\uncroped"
    slidingWindowPatchDir(imgDir = masksDirPath, 
                          imgMode='L',
                          patch_size = (512, 512),
                          stride = (128, 128),
                          save_dir=r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\masks",
                          visualize=False
    )

    # ---> MAKING DATASET SPLIT 
    splitDatasetInDirs(
        trainSamplesCounts=80,
        testSamplesCounts=20,
        sourceImgDir=r"M:\train-dataset-stride-128\images",
        sourceMasksDir=r"M:\train-dataset-stride-128\masks",
        sourceColoredMasksDir = r"M:\train-dataset-stride-128\masks\croped-colored",
        outputBaseDir=r"M:\train-dataset-stride-128\train_80"
    )
    trainDatasetPath = r"M:\train-dataset-stride-128\train_80"

    # ---> MODEL CONFIGURATION
    with open(r"C:\Users\Вика\YandexDisk-pawlova12\WORK\BIOFILMS\unet-project\unet-project-torch\config\train_config.json") as f:
        cfg = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainDataset = BiofilmDataset(cfg["paths"]["images"],
                                   cfg["paths"]["masks"],
                                   cfg["paths"]["colored_masks"],
                                   mode="train")

    trainLoader = DataLoader(trainDataset, batch_size=cfg["train"]["batch_size"], shuffle=True)

    model = build_model(cfg).to(device)
    
    # ---> LEARNING PROCESS
    trainModel(model, trainLoader, device, cfg)

    # ---> TESTING MODEL
    with open(r"C:\Users\Вика\YandexDisk-pawlova12\WORK\BIOFILMS\unet-project\unet-project-torch\config\train_config.json") as f:
        cfg = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = BiofilmDataset(
        cfg["paths"]["images"],
        cfg["paths"]["masks"],
        cfg["paths"]["colored_masks"],
        mode="test"
    )
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg["train"]["batch_size"],
                             shuffle=False)

    model = build_model(cfg["model"]).to(device)
    #load_checkpoint(model, "/home/VizaVi/unet-project-torch/checkpoints/model_epoch_30.pth")
    model.eval()

    test_model(model, test_loader, test_dataset, device, visualize=True, max_vis=30)
    
    # ---> TEST ONE SEM-IMAGE
    imagePath = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\testSourceDir\images"
    maskPath = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\testSourceDir\masks-binarized"
    coloredMaskPath = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\dataset\testSourceDir\masks"
    configPath = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\unet-project-torch\config\test_config.json"
    checkpointPath = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project\final_model_epoch_500.pth"
    testOneSEMimage(imgDir = imagePath,
                       maskDir = maskPath,
                       coloredMaskDir = coloredMaskPath,
                       outputDir = r"C:\Users\Victory\YandexDisk\WORK\BIOFILMS\unet-project",
                       configPath = configPath,
                       checkpointPath = checkpointPath
    )


if __name__ == "__main__":
    main()
