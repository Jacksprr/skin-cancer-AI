import os
import pandas as pd

# AUTO PATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

TRAIN_DIR = os.path.join(PROJECT_ROOT, "datasets", "combined_split", "train")
METADATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "metadata_combined_balanced.csv")

OUTPUT_METADATA = os.path.join(PROJECT_ROOT, "datasets", "metadata_train_only.csv")

print("Filtering metadata to match TRAIN images...")

metadata = pd.read_csv(METADATA_PATH)

# collect train image ids
train_ids = []

for cls in os.listdir(TRAIN_DIR):

    class_path = os.path.join(TRAIN_DIR, cls)

    if not os.path.isdir(class_path):
        continue

    for file in os.listdir(class_path):
        if file.lower().endswith(".jpg"):
            train_ids.append(file.replace(".jpg",""))

train_ids = set(train_ids)

# filter metadata
filtered_metadata = metadata[metadata["image_id"].astype(str).isin(train_ids)]

print("Original metadata:", len(metadata))
print("Filtered metadata:", len(filtered_metadata))

filtered_metadata.to_csv(OUTPUT_METADATA, index=False)

print("✅ Saved clean metadata:", OUTPUT_METADATA)