# src/model.py
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch

def build_model(cfg):
    model = smp.UnetPlusPlus(
        encoder_name=cfg.get('encoder_name', 'resnet34'),
        encoder_weights=cfg.get('encoder_weights', 'imagenet'),
        in_channels=1,  # grayscale
        classes=2,
        activation=cfg.get('activation_function', None)
    )
    return model

