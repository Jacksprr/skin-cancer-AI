import os
import pandas as pd

# =====================
# PATHS
# =====================
metadata_path = "datasets/metadata_combined.csv"
split_base = "datasets/combined_split"

train_folder = os.path.join(split_base, "train")
val_folder = os.path.join(split_base, "val")

# =====================
# LOAD METADATA
# =====================
meta = pd.read_csv(metadata_path)

# Remove NaN labels just in case
meta = meta[meta["dx"].notnull()].reset_index(drop=True)

# =====================
# GET IMAGE IDS FROM FOLDERS
# =====================
train_ids = {f.replace(".jpg","") for f in os.listdir(train_folder) if f.endswith(".jpg")}
val_ids = {f.replace(".jpg","") for f in os.listdir(val_folder) if f.endswith(".jpg")}

# =====================
# SPLIT METADATA
# =====================
train_meta = meta[meta["image_id"].isin(train_ids)]
val_meta = meta[meta["image_id"].isin(val_ids)]

# =====================
# SAVE
# =====================
train_meta.to_csv("datasets/combined_train.csv", index=False)
val_meta.to_csv("datasets/combined_val.csv", index=False)

print("Combined metadata split created ✅")
print("Train samples:", len(train_meta))
print("Val samples:", len(val_meta))