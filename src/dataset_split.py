# src/dataset_split.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import json

def is_valid_file(path):
    return os.path.isfile(path) and not os.path.basename(path).startswith('.')

def get_strata(ratio):
    if ratio == 0.0:
        return 0  # только фон
    elif ratio < 0.05:
        return 1  # очень мало биоплёнки
    elif ratio < 0.15:
        return 2
    elif ratio < 0.5:
        return 3
    else:
        return 4  # много биоплёнки

def calculate_biofilm_ratio(mask_path):
    mask = np.array(Image.open(mask_path).convert("L"))
    return np.mean(mask > 0) 

def stratify_split(images_dir, masks_dir, output_path):
    image_files = sorted([f for f in os.listdir(images_dir) if is_valid_file(os.path.join(images_dir, f))])
    mask_files = sorted([f for f in os.listdir(masks_dir) if is_valid_file(os.path.join(masks_dir, f))])
    
    print(f"Found {len(image_files)} images and {len(mask_files)} masks")
    
    assert len(image_files) == len(mask_files)

    image_paths = [os.path.join(images_dir, f) for f in image_files]
    mask_paths = [os.path.join(masks_dir, f) for f in mask_files]

    ratios = [calculate_biofilm_ratio(m) for m in mask_paths]
    strata = [get_strata(r) for r in ratios]

    
    def show_strata_distribution(strata, name="all"):
        plt.hist(strata, bins=np.arange(6)-0.5, edgecolor='black')
        plt.title(f"Стратификация{name}")
        plt.xlabel("Страта (0 = фон, 1 = мало, 4 = много)")
        plt.ylabel("Кол-во масок")
        plt.xticks(range(5))
        plt.grid(True)
        plt.show()

    show_strata_distribution(strata, "dataset")
    
    img_train, img_val_test, mask_train, mask_val_test, strata_train, strata_val_test = train_test_split(
        image_paths, mask_paths, strata, test_size=0.3, stratify=strata, random_state=42
    )
    
    strata_val = [get_strata(calculate_biofilm_ratio(p)) for p in mask_val_test]
    img_val, img_test, mask_val, mask_test = train_test_split(
        img_val_test, mask_val_test, test_size=0.5, stratify=strata_val, random_state=42
    )

    split = {
        "train": {"images": img_train, "masks": mask_train},
        "val": {"images": img_val, "masks": mask_val},
        "test": {"images": img_test, "masks": mask_test},
    }

    with open(output_path, "w") as f:
        json.dump(split, f, indent=4)

    print("Dataset split saved to:", output_path)
    for part in split:
        print(f"{part}: {len(split[part]['images'])} samples")

