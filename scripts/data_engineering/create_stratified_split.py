import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# PATHS
# =========================
output_dir = "datasets/combined_split"
metadata_path = "datasets/metadata_combined.csv"

# =========================
# LOAD METADATA
# =========================
meta = pd.read_csv(metadata_path)

# Remove rows with missing labels
meta = meta[meta["dx"].notnull()].reset_index(drop=True)

# Remove duplicate images
meta = meta.drop_duplicates(subset="image_id").reset_index(drop=True)

# ⚠️ DO NOT filter by image_path (this was breaking dataset)

# Shuffle dataset
meta = meta.sample(frac=1, random_state=42).reset_index(drop=True)

print("Total combined samples used:", len(meta))

# =========================
# STRATIFIED SPLIT
# =========================
train_meta, val_meta = train_test_split(
    meta,
    test_size=0.2,
    stratify=meta["dx"],
    random_state=42
)

print("\nTrain distribution:")
print(train_meta["dx"].value_counts())

print("\nValidation distribution:")
print(val_meta["dx"].value_counts())

# =========================
# RESET OUTPUT FOLDER
# =========================
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)

# =========================
# COPY IMAGES INTO CLASS FOLDERS
# =========================
def copy_images(df, split):

    for _, row in df.iterrows():

        src = row["image_path"]

        class_dir = os.path.join(output_dir, split, row["dx"])
        os.makedirs(class_dir, exist_ok=True)

        dst = os.path.join(class_dir, row["image_id"] + ".jpg")

        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print("Missing image:", src)

print("\nCopying TRAIN images...")
copy_images(train_meta, "train")

print("\nCopying VAL images...")
copy_images(val_meta, "val")

print("\nCombined stratified split completed successfully ✅")