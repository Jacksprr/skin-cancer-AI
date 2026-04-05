import os
import pandas as pd

# AUTO PATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

TRAIN_DIR = os.path.join(PROJECT_ROOT, "datasets", "combined_split", "train")
METADATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "metadata_combined_balanced.csv")

print("Checking dataset integrity...")

metadata = pd.read_csv(METADATA_PATH)

metadata_ids = set(metadata["image_id"].astype(str))

folder_ids = set()

class_names = []

for cls in os.listdir(TRAIN_DIR):

    class_path = os.path.join(TRAIN_DIR, cls)

    if not os.path.isdir(class_path):
        continue

    class_names.append(cls)

    for file in os.listdir(class_path):

        if file.lower().endswith(".jpg"):
            image_id = file.replace(".jpg","")
            folder_ids.add(image_id)

# =====================
# CHECKS
# =====================

missing_in_csv = folder_ids - metadata_ids
missing_in_folder = metadata_ids - folder_ids

print("\nRESULTS:")

print(f"Images in folders: {len(folder_ids)}")
print(f"Images in metadata: {len(metadata_ids)}")

if len(missing_in_csv)==0:
    print("✅ No images missing in CSV")
else:
    print("❌ Images without metadata:", len(missing_in_csv))

if len(missing_in_folder)==0:
    print("✅ No metadata without image")
else:
    print("❌ Metadata rows without image:", len(missing_in_folder))

# CLASS CHECK
csv_classes = set(metadata["dx"].unique())

if csv_classes == set(class_names):
    print("✅ Class labels match folders")
else:
    print("❌ Class mismatch detected")

print("\nVerification finished.")