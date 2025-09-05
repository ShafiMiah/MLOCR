from functools import cache
from ultralytics import YOLO
from pathlib import Path
import torch
import cv2
import os
import Extensions.MLMSettings as MLMSetting
import easyocr
import re
def TrainML():
    # Paths
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(torch.cuda.is_available())
    # print(torch.cuda.get_device_name(0))
    training_model_weight  = Path.cwd() /  "Train.pt"
    if os.path.exists(r'runs\detect\train\weights\best.pt'):
        training_model_weight = r'runs\detect\train\weights\best.pt'
    elif not os.path.exists(training_model_weight):
         training_model_weight = r'yolov8s.pt'
    else:
         training_model_weight = r'yolov8models\small.yaml'
    NumberOfEpoch = 100
    if not MLMSetting.NUMBER_OF_EPOCH == None:
        NumberOfEpoch = int(MLMSetting.NUMBER_OF_EPOCH)
    model = YOLO(training_model_weight)
    model.to(DEVICE)
    # ModelParameters.init_seed(YOLO('model.yaml'), "WeightPath/"+str(0)+"/seed.pkl")
    model.train(data='dataset.yaml', epochs = NumberOfEpoch,imgsz=1280,batch=16,workers=8, device = 0 if torch.cuda.is_available() else 'cpu',resume = True, cache= True)

def predictImageObjectYOLO(image_path, label_directory=None, class_directory=None):
    use_gpu = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(r'runs\detect\train\weights\best.pt')
    model.to(DEVICE)

    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]
    results = model.predict(img)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    if label_directory is not None:
        MLMSetting.TRAIN_LABEL_DIRECTORY = label_directory

    if class_directory is not None:
        MLMSetting.CLASS_FILE_DIRECORY = class_directory
    else:
        MLMSetting.CLASS_FILE_DIRECORY = MLMSetting.get_or_create_setting(
            "config/settings.xml", 
            "ClassFileDirectory", 
            r"SMLabel\data\predefined_classes.txt"
        )

    os.makedirs(MLMSetting.TRAIN_LABEL_DIRECTORY, exist_ok=True)
    label_path = os.path.join(MLMSetting.TRAIN_LABEL_DIRECTORY, base_name + ".txt")

    reader = easyocr.Reader(['en'], gpu=use_gpu)
    allowlist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    lines = []
    class_names = []
    class_id = 0

    # Compile regex if available
    regex_pattern = MLMSetting.REGEX.strip()
    regex = re.compile(regex_pattern) if regex_pattern else None

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Crop ROI
            roi = img[y1:y2, x1:x2]

            # OCR
            text_result = reader.readtext(roi)
            text = " ".join([res[1] for res in text_result]).strip()
            if not text:
                continue

            # Apply regex filter (if defined)
            if regex:
                match = regex.search(text)
                if not match:
                    continue
                text = match.group(0)  # keep matched substring

            # Save class name
            class_names.append(text)

            # Convert box to YOLO format
            xc, yc, w, h = voc_to_yolo(x1, y1, x2, y2, img_w, img_h)
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            class_id += 1

    # Save YOLO label file
    with open(label_path, "w") as f:
        f.write("\n".join(lines))

    # Ensure class directory exists
    os.makedirs(os.path.dirname(MLMSetting.CLASS_FILE_DIRECORY), exist_ok=True)

    # Save class names
    class_txt_path = MLMSetting.CLASS_FILE_DIRECORY
    with open(class_txt_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")

def voc_to_yolo(x1, y1, x2, y2, img_w, img_h):
    x_center = (x1 + x2) / 2.0 / img_w
    y_center = (y1 + y2) / 2.0 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height


