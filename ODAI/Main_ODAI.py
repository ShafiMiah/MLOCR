import easyocr
import re
import cv2
import torch
import DataPreProcessor
import OCRToYoloDataset
import TrainML
import subprocess
from pathlib import Path
import shutil
import SMLabel.MLMAnnotation as MLAnnotation
import os
import Extensions.MLMSettings as MLMSetting
import sys
import argparse
import yaml
# -----------------------------
# Configuration
# -----------------------------
def CreateConfiguration():
    val_images = Path(os.path.dirname(MLMSetting.TRAIN_IMAGE_DIRECTORY)) / "val"

    data = {
        "nc": 1,
        "train": MLMSetting.TRAIN_LABEL_DIRECTORY,
        "val": str(val_images),  # convert to string if needed
        "names": {0: "Text"}
    }

    # Path to the YAML file
    file_path = Path.cwd() / "dataset.yaml"

    # Write YAML file
    with open(file_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def SingleImageAnnotation(image_path = None, label_path = None, class_file = None):
     MLAnnotation.run_training_main(image_location = image_path,label_location= label_path, class_file= class_file)

def PredictHotspotImages(image_path = None, display_in_viewer = False):
     base_name = os.path.splitext(os.path.basename(image_path))[0]
     label_data_path = MLMSetting.OUTPUT_DIRECORY
     class_file = os.path.join(label_data_path, base_name + "class.txt")
     TrainML.predictImageObjectYOLO(image_path, label_data_path, class_file)
     if display_in_viewer == True:
        MLAnnotation.run_training_main(image_location = image_path,label_location= label_data_path, class_file= class_file)

def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")

def Initialize():
    if not os.path.exists(MLMSetting.TRAIN_LABEL_DIRECTORY):
        os.makedirs(MLMSetting.TRAIN_LABEL_DIRECTORY)
    if not os.path.exists(r'runs\detect\train\weights'):
        os.makedirs(r'runs\detect\train\weights')
    if not os.path.exists(os.path.dirname(MLMSetting.CLASS_FILE_DIRECORY)):
        os.makedirs(os.path.dirname(MLMSetting.CLASS_FILE_DIRECORY), exist_ok=True)

    yamlFileLocation =  Path.cwd() /  "dataset.yaml"
    if not os.path.exists(MLMSetting.CLASS_FILE_DIRECORY):
        with open(MLMSetting.CLASS_FILE_DIRECORY, "w") as f:
            f.write("Text")
    if not os.path.exists(yamlFileLocation):
        CreateConfiguration()
# -----------------------------
# Run
# -----------------------------

def get_main_app(argv=None):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    if not argv:
        argv = []
    # Tzutalin 201705+: Accept extra agruments to change predefined class file
    Initialize()
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--operation",  required=True, help="Type of operation: TransferCSVImage,AutoAnnotation,PredictHotspot")
    argparser.add_argument("--imagepath",  help="This is the source image path")
    argparser.add_argument("--trainpath",  help="This is the train image path")
    argparser.add_argument("--labelpath",  help="This is the train annotation path")
    argparser.add_argument("--classfile",  help="This is the train annotation path")
    argparser.add_argument("--imagenamescontainerfile",  help="This is contains all the names of images which used for training")
    argparser.add_argument("--outputpath",  help="This is the path where the prediction annotationa dn class will be written")
    argparser.add_argument("--regex",  help="This will filter out the text with regex")
    argparser.add_argument(
        "--display",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Set True or False to show/annotate on viewer"
    )

    args = argparser.parse_args(argv[1:])

    if args.imagepath:
        MLMSetting.SOURCE_IMAGE_DIRECTORY = args.imagepath
    if args.trainpath:
        MLMSetting.TRAIN_IMAGE_DIRECTORY = args.trainpath
    if args.labelpath:
        MLMSetting.TRAIN_LABEL_DIRECTORY = args.labelpath
    if args.classfile:
        MLMSetting.CLASS_FILE_DIRECORY = args.classfile
    if args.imagenamescontainerfile:
        MLMSetting.ILLUSTRATION_IMAGE_NAMES_CONTAINER_FILE = args.imagenamescontainerfile
    if args.outputpath:
        MLMSetting.OUTPUT_DIRECORY = args.outputpath
    if args.regex:
        MLMSetting.REGEX = args.regex
    match args.operation:
        case "TransferCSVImage":
            #Step 1: This will copy images from one directory to another. Read the images from a csv file.
            # python Main_ODAI.py --operation TransferCSVImage --imagepath "C:\SourceImages" --trainpath "SMLabel\data\images\train" --imagenamescontainerfile "c:\imagenames.csv"
            DataPreProcessor.GetData()
        case "AutoAnnotation":
            # Step 2: This will generate YOLO model dataset with automatically annotate label using OCS
            # python Main_ODAI.py --operation AutoAnnotation  --trainpath "SMLabel\data\images\train" --labelpath "SMLabel\data\labels\train" --classfile "SMLabel\data\predefined_classes.txt"
            OCRToYoloDataset.YOLODataPreProcessor()
        case "ManualAnnotation":
            #Step 3: This will generate YOLO model dataset and you can annotate label manually
            # python Main_ODAI.py --operation ManualAnnotation  --trainpath "SMLabel\data\images\train" --labelpath "SMLabel\data\labels\train" --classfile "SMLabel\data\predefined_classes.txt"
            MLAnnotation.run_training_main()
        case "CreatevalidationSet":
            #Step 4: This will generate YOLO model validation dataset
            # python Main_ODAI.py --operation CreatevalidationSet  --trainpath "SMLabel\data\images\train" --labelpath "SMLabel\data\labels\train"
            OCRToYoloDataset.TrainVal()
        case "TrainML":
            #Step 5: This will start training the model
            # python Main_ODAI.py --operation TrainML  --trainpath "SMLabel\data\images\train" --labelpath "SMLabel\data\labels\train" --classfile "SMLabel\data\predefined_classes.txt"
            TrainML.TrainML()
        case "PredictHotspot":
           #Step 6: This will predict the hotspot
           # python Main_ODAI.py --operation PredictHotspot  --imagepath "C:\SourceImages\imagetopredict.jpg" --display
           PredictHotspotImages(args.imagepath,args.display)
        case "SingleImageAnnotation":
           #Step 6: This will predict the hotspot
           # python Main_ODAI.py --operation SingleImageAnnotation  --imagepath "C:\SourceImages\imagetopredict.jpg" --labelpath "SMLabel\data\labels\train" --classfile "SMLabel\data\predefined_classes.txt"
           SingleImageAnnotation(args.imagepath,args.display)
def main():
    """construct main app and run it"""
    get_main_app(sys.argv)

if __name__ == '__main__':
   
    main()