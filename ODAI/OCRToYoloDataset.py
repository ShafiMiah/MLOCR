from ast import Import
import os
import cv2
import easyocr
import torch
import re
import glob
import random
import shutil
from pathlib import Path
from PIL import Image
import Extensions.MLMSettings as MLMSetting
# === Paths ===
image_dir = MLMSetting.TRAIN_IMAGE_DIRECTORY          # input folder with images
labels_dir =  MLMSetting.TRAIN_LABEL_DIRECTORY           # output folder for YOLO txt files
CONF_THRESH = 0.8  
os.makedirs(labels_dir, exist_ok=True)
use_gpu = torch.cuda.is_available()
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def YOLODataPreProcessor():
# === EasyOCR setup ===
    reader = easyocr.Reader(['en'],  gpu=use_gpu)
    allowlist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # === Process each image ===
    for img_name in sorted(os.listdir(image_dir), key=natural_sort_key):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")

        # Read image
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Run OCR
        results = reader.readtext(img_path, detail=1, paragraph=False, allowlist=allowlist)

        lines = []
        for bbox, text, conf in results:
             if conf >= CONF_THRESH:
                # Convert OCR bbox -> YOLO format
                x_min = min(pt[0] for pt in bbox)
                y_min = min(pt[1] for pt in bbox)
                x_max = max(pt[0] for pt in bbox)
                y_max = max(pt[1] for pt in bbox)

                # Normalize
                x_center = ((x_min + x_max) / 2) / w
                y_center = ((y_min + y_max) / 2) / h
                width = (x_max - x_min) / w
                height = (y_max - y_min) / h

                # Class 0 = "text"
                lines.append(f"0 {x_center} {y_center} {width} {height}")

        # Save YOLO annotation file
        if lines:
            print("Label: " +os.path.splitext(img_name)[0])
            with open(label_path, "w") as f:
                f.write("\n".join(lines))

def RemoveSvgFile():
    for file in glob.glob(os.path.join(image_dir, "*.svg")):
        os.remove(file)
        print(f"Deleted: {file}")
def TrainVal():
    images_path =  MLMSetting.TRAIN_IMAGE_DIRECTORY
    labels_path =  MLMSetting.TRAIN_LABEL_DIRECTORY  # optional if you have labels

    val_images = os.path.dirname(MLMSetting.TRAIN_IMAGE_DIRECTORY) / "val"
    val_labels = os.path.dirname(MLMSetting.TRAIN_LABEL_DIRECTORY)  / "val"

    # Create val folders
    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)

    # Read all image files (jpg, jpeg, png)
    all_images = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        all_images.extend(images_path.glob(ext))
    all_images = sorted(all_images)  # deterministic order

    # Take 20% for validation
    val_count = int(0.2 * len(all_images))
    random.seed(42)  # fixed seed for reproducibility
    val_files = random.sample(all_images, val_count)

    # Copy files to val folder
    for img_path in val_files:
        shutil.copy(img_path, val_images / img_path.name)
    
        # Copy corresponding label if exists
        label_path = labels_path / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy(label_path, val_labels / f"{img_path.stem}.txt")

    print(f"Validation set created with {len(val_files)} images")

