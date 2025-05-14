import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

def iou_score(outputs, targets, smooth=1e-6):
    preds = torch.argmax(outputs, dim=1)
    intersection = ((preds == 1) & (targets == 1)).float().sum((1, 2))
    union = ((preds == 1) | (targets == 1)).float().sum((1, 2))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

def dice_score(outputs, targets, smooth=1e-6):
    preds = torch.argmax(outputs, dim=1)
    intersection = ((preds == 1) & (targets == 1)).float().sum((1, 2))
    total = preds.sum((1, 2)) + targets.sum((1, 2))
    dice = (2.0 * intersection + smooth) / (total + smooth)
    return dice.mean().item()

def test_model(model, test_loader, test_dataset, device, visualize=True, max_vis=5):
    model.eval()
    total_iou = 0.0
    total_dice = 0.0
    count = 0
    vis_count = 0

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)

            iou = iou_score(outputs, masks)
            dice = dice_score(outputs, masks)

            print(f"[{i}] IoU: {iou:.4f} | Dice: {dice:.4f}")
            total_iou += iou
            total_dice += dice
            count += 1

            if visualize and vis_count < max_vis:
                img_path = test_dataset.image_paths[i]
                img_name = os.path.basename(img_path)

                orig_img = np.array(Image.open(img_path).convert("L"))
                orig_color_mask = np.array(Image.open(test_dataset.colored_mask_paths[i]).convert("RGB"))
                binary_mask = masks[0].cpu().numpy()

                pred_mask = torch.argmax(outputs[0], dim=0).cpu().numpy()

                fig, axs = plt.subplots(1, 4, figsize=(18, 4))
                axs[0].imshow(orig_img, cmap="gray")
                axs[0].set_title(f"{img_name}", fontsize=10)
                axs[1].imshow(orig_color_mask)
                axs[1].set_title("Original mask (RGB)")
                axs[2].imshow(binary_mask, cmap="gray")
                axs[2].set_title("Binary mask")
                axs[3].imshow(pred_mask, cmap="gray")
                axs[3].set_title("Prediction")
                for ax in axs:
                    ax.axis("off")
                plt.tight_layout()
                plt.show()

                vis_count += 1

    print(f"\n=== TOTAL IoU: {total_iou / count:.4f} | Dice: {total_dice / count:.4f} ===")
