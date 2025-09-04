import os
import xml.etree.ElementTree as ET


def get_or_create_setting(xml_path: str, key: str, default_value: str) -> str:
    # Ensure directory exists
    os.makedirs(os.path.dirname(xml_path), exist_ok=True)

    if os.path.exists(xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            if root.tag != "settings":
                root = ET.Element("settings")
                tree = ET.ElementTree(root)
        except ET.ParseError:
            # Corrupted XML -> recreate
            root = ET.Element("settings")
            tree = ET.ElementTree(root)
    else:
        root = ET.Element("settings")
        tree = ET.ElementTree(root)

    # Look for existing node
    for node in root.findall("add"):
        if node.get("key") == key:
            return node.get("value", default_value)

    # If not found → create new node
    node = ET.Element("add", key=key, value=default_value)
    root.append(node)

    # Save back to file
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    return default_value


def set_setting(xml_path: str, key: str, value: str) -> None:
    os.makedirs(os.path.dirname(xml_path), exist_ok=True)

    if os.path.exists(xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            if root.tag != "settings":
                root = ET.Element("settings")
                tree = ET.ElementTree(root)
        except ET.ParseError:
            root = ET.Element("settings")
            tree = ET.ElementTree(root)
    else:
        root = ET.Element("settings")
        tree = ET.ElementTree(root)

    # Find node or create new one
    for node in root.findall("add"):
        if node.get("key") == key:
            node.set("value", value)
            break
    else:
        node = ET.Element("add", key=key, value=value)
        root.append(node)

    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

SOURCE_IMAGE_DIRECTORY = get_or_create_setting("config/settings.xml", "SourceImageDirectory", r"C:\Data\SourceImages")
ILLUSTRATION_IMAGE_NAMES_CONTAINER_FILE = get_or_create_setting("config/settings.xml", "IllustrationImageNamesContainerFile", r"C:\Data\images.csv")
TRAIN_IMAGE_DIRECTORY = get_or_create_setting("config/settings.xml", "TrainImageDirectory", r"C:\Data\images\train")
TRAIN_LABEL_DIRECTORY = get_or_create_setting("config/settings.xml", "LabelDirectory", r"C:\Data\labels\train")
CLASS_FILE_DIRECORY = get_or_create_setting("config/settings.xml", "ClassFileDirectory", r"SMLabel\data\predefined_classes.txt")
OUTPUT_DIRECORY = get_or_create_setting("config/settings.xml", "OutputDirectory", r"C:\Temp")
REGEX = get_or_create_setting("config/settings.xml", "Regex", "")
NUMBER_OF_EPOCH = get_or_create_setting("config/settings.xml", "NumberOfEpoch", "100")
