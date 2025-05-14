# main.py
import json
from PIL import Image
import torch
import os
from numpy import save
from torch.utils.data import DataLoader
from src.dataset import BiofilmDataset, splitDatasetInDirs
from src.model import build_model
from src.train import trainModel
from src.preprocessing import binarizeMaskDir, cropLineBelow, slidingWindowPatchDir
from src.test import test_model

def main():
    # ---> TESTING MODEL
    with open(r"C:\Users\Вика\YandexDisk-pawlova12\РАБОТА\BIOFILMS\unet-project\unet-project-torch\config\train_config.json") as f:
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
    #model.eval()

    test_model(model, test_loader, test_dataset, device, visualize=True, max_vis=30)


def mainGlobal():
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
    masksDirPath = r"C:\Users\Вика\YandexDisk-pawlova12\РАБОТА\BIOFILMS\unet-project\unet-project-torch\dataset\raw\masks\BSE"
    
    binarizeMaskDir(
    masksDirPath,
    "RGB",
    masksList,
    targetClassColor=(36, 179, 83),
    secondClassColor=(0, 0, 0),
    saveDir=r"C:\Users\Вика\YandexDisk-pawlova12\РАБОТА\BIOFILMS\unet-project\unet-project-torch\dataset\masks\uncroped",
    vizN=5
    )

    # ---> CROPPING IMAGES AND MASKS
    """ 
    Croping black line below,
    cutting img and masks with sliding window
    """
    imagesDirPath = r"C:\Users\Вика\YandexDisk-pawlova12\РАБОТА\BIOFILMS\unet-project\unet-project-torch\dataset\raw\images\BSE"
    slidingWindowPatchDir(imgDir = imagesDirPath, 
                          imgMode='L',
                          patch_size = (512, 512),
                          stride = (256, 256),
                          save_dir=r"C:\Users\Вика\YandexDisk-pawlova12\РАБОТА\BIOFILMS\unet-project\unet-project-torch\dataset\images",
                          visualize=False
    )

    coloredMasksDirPath = r"C:\Users\Вика\YandexDisk-pawlova12\РАБОТА\BIOFILMS\unet-project\unet-project-torch\dataset\raw\masks\BSE"
    slidingWindowPatchDir(imgDir = coloredMasksDirPath, 
                          imgMode='RGB',
                          patch_size = (512, 512),
                          stride = (256, 256),
                          save_dir=r"C:\Users\Вика\YandexDisk-pawlova12\РАБОТА\BIOFILMS\unet-project\unet-project-torch\dataset\masks\croped-colored",
                          visualize=False
    )

    masksDirPath = r"C:\Users\Вика\YandexDisk-pawlova12\РАБОТА\BIOFILMS\unet-project\unet-project-torch\dataset\masks\uncroped"
    slidingWindowPatchDir(imgDir = masksDirPath, 
                          imgMode='L',
                          patch_size = (512, 512),
                          stride = (256, 256),
                          save_dir=r"C:\Users\Вика\YandexDisk-pawlova12\РАБОТА\BIOFILMS\unet-project\unet-project-torch\dataset\masks",
                          visualize=False
    )

    # ---> MAKING DATASET SPLIT 
    splitDatasetInDirs(
    trainSamplesCounts=80,
    testSamplesCounts=20,
    sourceImgDir=r"C:\Users\Вика\YandexDisk-pawlova12\РАБОТА\BIOFILMS\unet-project\unet-project-torch\dataset\images",
    sourceMasksDir=r"C:\Users\Вика\YandexDisk-pawlova12\РАБОТА\BIOFILMS\unet-project\unet-project-torch\dataset\masks",
    outputBaseDir=r"C:\Users\Вика\YandexDisk-pawlova12\РАБОТА\BIOFILMS\unet-project\unet-project-torch\train-dataset"
    )
    trainDatasetPath = r"C:\Users\Вика\YandexDisk-pawlova12\РАБОТА\BIOFILMS\unet-project\unet-project-torch\train-dataset\train_80"

    # ---> MODEL CONFIGURATION
    with open(r"C:\Users\Вика\YandexDisk-pawlova12\РАБОТА\BIOFILMS\unet-project\unet-project-torch\config\train_config.json") as f:
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
    with open(r"C:\Users\Вика\YandexDisk-pawlova12\РАБОТА\BIOFILMS\unet-project\unet-project-torch\config\train_config.json") as f:
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
    #model.eval()

    test_model(model, test_loader, device, visualize=True, max_vis=30)

if __name__ == "__main__":
    main()
