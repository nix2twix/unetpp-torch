# src/dataset_split.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from PIL import Image

def isValidFile(path):
    # ---> Игнорирование файлов вида .temp
    return os.path.isfile(path) and not os.path.basename(path).startswith('.')

def getRange(imgPIL):
    # ---> Вывод диапазона значений для каждого канала изображения
    img_np = np.array(imgPIL)
    if img_np.ndim == 3 and img_np.shape[2] == 3:
        rgbRange = {}
        for i, channel in enumerate(['R', 'G', 'B']):
                    min_val = img_np[:, :, i].min()
                    max_val = img_np[:, :, i].max()
                    print(f"{channel}-channel: [{min_val}, {max_val}]")
                    rgbRange[channel] = (min_val, max_val)
        return rgbRange
    else:
        min_val = img_np.min()
        max_val = img_np.max()
        print(f"Grayscale: [{min_val}, {max_val}]")
        grayscaleRange = (min_val, max_val)
        return grayscaleRange

def countUniqueMaskColorDir(path, mode='L', masksList = None):
    # ---> Обработка указанной директории path
    for f in os.scandir(path):
        countUniqueMaskColor(f, mode, masksList)

def countUniqueMaskColor(maskPath, mode='L', masksList = None):
    # ---> Обработка одного файла maskPath
    if isValidFile(maskPath):
        with Image.open(maskPath) as img:
            imgInMode = img.convert(mode)
            print(f"The masks {os.path.basename(maskPath)} [{imgInMode.size[0]} x {imgInMode.size[1]}]" 
                    + f" ({imgInMode.size[0] * imgInMode.size[1]} pixels) consist of: ")
            # ---> Автоопределение классов при наличии словаря классов 
            if (masksList != None):
                for color in imgInMode.getcolors():
                    print(f"{color[0]} pixels of {color[1]} color, " 
                        + f"the class is {masksList[color[1]]}")
                
            # ---> Просто проверка кол-ва пикселей и их уникальных цветов
            else:
                for color in imgInMode.getcolors():
                    print(f"{color[0]} pixels of {color[1]} color")

def countPixelsWithColor(maskNP, targetRGB):
    r, g, b = targetRGB
    maskMatch = (maskNP[:, :, 0] == r) & (maskNP[:, :, 1] == g) & (maskNP[:, :, 2] == b)
    return np.sum(maskMatch)

def binarizeMaskDir(maskPath, maskMode, masksList, targetClassColor, secondClassColor = "Background", saveDir = None, vizN = 0):
    if os.path.isdir(maskPath):
        maskFiles = sorted([
            os.path.join(maskPath, fname) for fname in os.listdir(maskPath)
            if fname.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        for i, path in enumerate(maskFiles):
            binarizeMask(path, maskMode, masksList, targetClassColor, secondClassColor, saveDir, vizN, idx=i)
    else:
        return binarizeMask(path, maskMode, masksList, targetClassColor, secondClassColor, saveDir, vizN, idx=0)

def binarizeMask(maskPath, maskMode, masksList, targetClassColor, secondClassColor = (0, 0, 0), saveDir = None, vizN = 0, idx=None):
    with Image.open(maskPath) as mask:
        originalMask = mask.convert(maskMode)

    maskNP = np.array(originalMask)
    print(f"The mask {os.path.basename(maskPath)} [{originalMask.size[0]} x {originalMask.size[1]}]" 
                + f" ({originalMask.size[0] * originalMask.size[1]} pixels)")
       
    # ---> Исходное изображение-маска
    print("Before binarization:") 
    getRange(originalMask)
    print(f"Pixels in class {masksList[targetClassColor]}: {countPixelsWithColor(maskNP, targetClassColor)}")
    countOtherClasses = 0
    for color in masksList:
        if (color != targetClassColor):
            countOtherClasses = countOtherClasses + countPixelsWithColor(maskNP, color)
            print(f"Pixels in class {masksList[color]}: {countPixelsWithColor(maskNP, color)}")
    print(f"Total pixels in not target classes {countOtherClasses}\n")

    # ---> Бинаризация
    targetRGB = next((rgb for rgb, name in masksList.items() if rgb == targetClassColor), None)
    bgRGB = next((rgb for rgb, name in masksList.items() if rgb == secondClassColor), None)

    if targetRGB is None:
        raise ValueError(f"Class '{targetClassColor}' not found in {maskPath}")
    if bgRGB is None:
        raise ValueError(f"Class '{secondClassColor}' not found in {maskPath}")

    binMask = np.all(maskNP == targetRGB, axis=-1).astype(np.uint8)

    print(f"\nAfter binarization:")
    getRange(binMask)
    print(f"Pixels in class ({targetClassColor}): {np.sum(binMask == 1)}")
    print(f"Pixels in class ({secondClassColor}): {np.sum(binMask == 0)}")
 
    if saveDir:
        os.makedirs(saveDir, exist_ok=True)
        save_path = os.path.join(saveDir, os.path.splitext(os.path.basename(maskPath))[0] + ".png")
        Image.fromarray(binMask * 255).save(save_path)

    if idx is not None and idx < vizN:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(originalMask)
        axs[0].set_title(f"{os.path.basename(maskPath)}")
        axs[1].imshow(binMask, cmap="gray")
        axs[1].set_title("Binary mask")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    return binMask

def cropLineBelow(imgPIL, countPx=120):
    width, height = imgPIL.size
    cropped_img = imgPIL.crop((0, 0, width, height - countPx))

    original_pixels = width * height
    cropped_pixels = width * (height - countPx)

    print(f"Original size: {width}x{height} = {original_pixels} pixels")
    print(f"Cropped size: {width}x{height - countPx} = {cropped_pixels} pixels")
    print(f"Removed pixels: {original_pixels - cropped_pixels} (Expected: {width * countPx})")

    return cropped_img

def slidingWindowPatch(imgPIL, img_name, patch_size=(512, 512), stride=(128, 128),
                       save_dir=None, visualize=True):
    imgPIL = cropLineBelow(imgPIL)
    img_np = np.array(imgPIL)
    img_height, img_width = img_np.shape[:2]

    patch_h, patch_w = patch_size
    stride_y, stride_x = stride

    base_name = os.path.splitext(os.path.basename(img_name))[0]
    os.makedirs(save_dir, exist_ok=True)

    patch_id = 0
    patch_list = []
    coords = []

    x_coords = list(range(0, img_width - patch_w + 1, stride_x))
    #x_coords.append(img_width - patch_w)

    y_coords = list(range(0, img_height - patch_h + 1, stride_y))
    #y_coords.append(img_height - patch_h)

    for y in y_coords:
        for x in x_coords:
            patch = imgPIL.crop((x, y, x + patch_w, y + patch_h))
            patch_path = os.path.join(save_dir, f"{base_name}.{patch_id:03d}.png")
            patch.save(patch_path)
            patch_list.append(patch)
            coords.append((x, y, patch_id))
            patch_id += 1

    print(f"Original image size: {img_width}x{img_height}")
    print(f"Patch size: {patch_w}x{patch_h}")
    print(f"Stride: ({stride_x}, {stride_y})")
    print(f"Total patches: {patch_id}")

    if visualize:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(imgPIL, cmap='gray')
        cmap = cm.get_cmap('hsv', len(coords))

        for i, (x, y, pid) in enumerate(coords):
            color = cmap(i)
            rect = patches.Rectangle((x, y), patch_w, patch_h,
                                        linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x + 5, y + 5, f"{pid}", color=color, fontsize=8, weight='bold')
        ax.set_title("Patch layout (color-coded)")
        ax.axis('off')
        plt.tight_layout()
        plt.show()

def slidingWindowPatchDir(imgDir, imgMode, patch_size=(512, 512), stride=(128, 128), save_dir=None, visualize=True):
    for imgName in os.scandir(imgDir):
        print(f"---> Current image: {imgName}")
        img = Image.open(imgName)
        img = img.convert(imgMode)
        slidingWindowPatch(img, imgName, patch_size, stride, save_dir, visualize)