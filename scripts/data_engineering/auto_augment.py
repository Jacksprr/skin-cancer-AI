import os
import cv2
import random
import pandas as pd
from tqdm import tqdm

# ===============================
# AUTO PATH HANDLING (IMPORTANT)
# ===============================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "combined_split", "train")
METADATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "metadata_combined.csv")
OUTPUT_METADATA = os.path.join(PROJECT_ROOT, "datasets", "metadata_combined_balanced.csv")

TARGET_COUNT = 400  # Target per class

print("Project root:", PROJECT_ROOT)
print("Training data:", DATA_DIR)
print("Metadata file:", METADATA_PATH)

# ===============================
# LOAD METADATA
# ===============================

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError("❌ metadata_combined.csv not found")

metadata = pd.read_csv(METADATA_PATH)

if "image_id" not in metadata.columns or "dx" not in metadata.columns:
    raise ValueError("❌ Metadata must contain 'image_id' and 'dx' columns")

# ===============================
# SAFE AUGMENTATION FUNCTION
# ===============================

def augment_image(img):
    h, w = img.shape[:2]

    # Horizontal flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)

    # Small rotation
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Brightness adjustment
    value = random.randint(-20, 20)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], value)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img


# ===============================
# AUGMENTATION LOOP
# ===============================

new_rows = []

for cls in os.listdir(DATA_DIR):

    class_path = os.path.join(DATA_DIR, cls)

    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path) if f.lower().endswith(".jpg")]
    current_count = len(images)

    print(f"\n📁 Class: {cls} | Current: {current_count}")

    if current_count >= TARGET_COUNT:
        print("   Already balanced. Skipping.")
        continue

    needed = TARGET_COUNT - current_count
    print(f"   Generating {needed} augmented images...")

    for i in tqdm(range(needed), desc=f"Augmenting {cls}"):

        base_img_name = random.choice(images)
        base_img_id = base_img_name.replace(".jpg", "")

        img_path = os.path.join(class_path, base_img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠ Could not read {img_path}")
            continue

        aug_img = augment_image(img)

        # Ensure unique ID
        unique_index = random.randint(100000, 999999)
        new_id = f"aug_{cls}_{unique_index}_{base_img_id}"
        new_filename = new_id + ".jpg"

        cv2.imwrite(os.path.join(class_path, new_filename), aug_img)

        # Duplicate metadata safely
        original_row = metadata[metadata["image_id"] == base_img_id]

        if original_row.empty:
            print(f"⚠ Metadata missing for {base_img_id}")
            continue

        row_copy = original_row.iloc[0].copy()
        row_copy["image_id"] = new_id
        new_rows.append(row_copy)


# ===============================
# SAVE UPDATED METADATA
# ===============================

if new_rows:
    new_metadata = pd.concat([metadata, pd.DataFrame(new_rows)], ignore_index=True)
    new_metadata.to_csv(OUTPUT_METADATA, index=False)
    print("\n✅ Augmentation complete.")
    print("Updated metadata saved to:", OUTPUT_METADATA)
else:
    print("\n⚠ No new images generated.")