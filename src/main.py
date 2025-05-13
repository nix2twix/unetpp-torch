# main.py
import json
import torch
from torch.utils.data import DataLoader
from src.dataset import BiofilmDataset
from src.model import build_model
from src.train import train_model

def main():
    # Загружаем конфиг
    with open(r"C:\Users\Victory\YandexDisk\РАБОТА\BIOFILMS\unet-project\unet-project-torch\config\train_config.json") as f:
        cfg = json.load(f)

    # Устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Датасет
    train_dataset = BiofilmDataset(cfg["paths"]["images"],
                                   cfg["paths"]["masks"],
                                   cfg["paths"]["split_json"],
                                   mode="train")

    train_loader = DataLoader(train_dataset, batch_size=cfg["train"]["batch_size"], shuffle=True)

    # Модель
    model = build_model(cfg).to(device)

    # Запуск обучения
    #train_model(model, train_loader, device, cfg)

if __name__ == "__main__":
    main()
