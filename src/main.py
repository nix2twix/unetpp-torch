# main.py
import json
from PIL import Image
#import torch
import os
from numpy import save
from torch.utils.data import DataLoader
from src.dataset import BiofilmDataset
from src.model import build_model
from src.train import train_model
from src.preprocessing import binarizeMaskDir, cropLineBelow, slidingWindowPatchDir

def main():
    # ---> MAKING DATASET SPLIT 
    """ 
    Croping black line below,
    cutting img and masks with sliding window
    """
    imagesDirPath = r"C:\Users\Вика\YandexDisk-pawlova12\РАБОТА\BIOFILMS\unet-project\unet-project-torch\dataset\masks\uncroped"
    slidingWindowPatchDir(imgDir = imagesDirPath, 
                          imgMode='L',
                          patch_size = (512, 512),
                          stride = (256, 256),
                          save_dir=r"C:\Users\Вика\YandexDisk-pawlova12\РАБОТА\BIOFILMS\unet-project\unet-project-torch\dataset\masks",
                          visualize=False
    )

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


    with open(r"C:\Users\Вика\YandexDisk-pawlova12\РАБОТА\BIOFILMS\unet-project\unet-project-torch\config\train_config.json") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Датасет
    train_dataset = BiofilmDataset(cfg["paths"]["images"],
                                   cfg["paths"]["masks"],
                                   mode="train")

    train_loader = DataLoader(train_dataset, batch_size=cfg["train"]["batch_size"], shuffle=True)

    # Модель
    model = build_model(cfg).to(device)

    # Запуск обучения
    #train_model(model, train_loader, device, cfg)

if __name__ == "__main__":
    main()
