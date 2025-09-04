import os
import shutil
import pandas as pd
import Extensions.MLMSettings as Settings

# Paths
image_names_csv_file = Settings.ILLUSTRATION_IMAGE_NAMES_CONTAINER_FILE       # CSV with a column of image names
source_dir = Settings.SOURCE_IMAGE_DIRECTORY     # folder with all images
dest_dir =  Settings.TRAIN_IMAGE_DIRECTORY       # destination folder
def GetData():

    os.makedirs(dest_dir, exist_ok=True)

    # Load CSV
    df = pd.read_csv(image_names_csv_file, header=None)

    # Get first column as list of filenames
    image_names = df[0].tolist()

    # Copy and rename
    for idx, img_name in enumerate(image_names):
        src_path = os.path.join(source_dir, img_name)
        if not os.path.exists(src_path):
            print(f"⚠️ Skipped: {img_name} not found in {source_dir}")
            continue

        ext = os.path.splitext(img_name)[1]
        dest_path = os.path.join(dest_dir, f"{idx}{ext}")
        shutil.copy(src_path, dest_path)
        print(f"Copied {img_name} → {dest_path}")

    print("✅ Done. All listed images copied and renamed.")
