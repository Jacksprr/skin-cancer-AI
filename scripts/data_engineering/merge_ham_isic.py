import pandas as pd
import os

# =========================
# LOAD HAM LABEL METADATA
# =========================
ham_label_path = "datasets/HAM10000_metadata.csv"
ham_labels = pd.read_csv(ham_label_path)

print("HAM labels:", len(ham_labels))

# =========================
# LOAD HAM PROCESSED FEATURES
# =========================
ham_feature_path = "datasets/metadata_processed.csv"
ham_features = pd.read_csv(ham_feature_path)

print("HAM features:", len(ham_features))

# =========================
# MERGE LABELS + FEATURES
# =========================
ham_df = pd.merge(
    ham_features,
    ham_labels[["image_id", "dx"]],
    on="image_id",
    how="left"
)

# =========================
# HAM IMAGE PATH (YOUR REAL LOCATION)
# =========================
ham_image_dir = "datasets/combined/"

def resolve_ham_path(image_id):

    p = os.path.join(ham_image_dir, image_id + ".jpg")

    if os.path.exists(p):
        return p
    else:
        return None

ham_df["image_path"] = ham_df["image_id"].apply(resolve_ham_path)

print("HAM merged samples:", len(ham_df))

# =========================
# LOAD ISIC CLEANED FILES
# =========================
isic_train_path = "datasets/isic_raw/isic2018_task3_train/isic_clean.csv"
isic_val_path = "datasets/isic_raw/isic2018_task3_val/isic_clean.csv"

isic_train_df = pd.read_csv(isic_train_path)
isic_val_df = pd.read_csv(isic_val_path)

isic_df = pd.concat([isic_train_df, isic_val_df], ignore_index=True)

print("ISIC samples:", len(isic_df))

# =========================
# FIX ISIC IMAGE PATHS
# =========================
train_image_path = "datasets/isic_raw/isic2018_task3_train/images/"
val_image_path = "datasets/isic_raw/isic2018_task3_val/images/"

def resolve_isic_path(image_id):

    p1 = os.path.join(train_image_path, image_id + ".jpg")
    p2 = os.path.join(val_image_path, image_id + ".jpg")

    if os.path.exists(p1):
        return p1
    elif os.path.exists(p2):
        return p2
    else:
        return None

isic_df["image_path"] = isic_df["image_id"].apply(resolve_isic_path)

isic_df = isic_df[isic_df["image_path"].notnull()].reset_index(drop=True)

print("Valid ISIC images:", len(isic_df))

# =========================
# ALIGN COLUMNS
# =========================
common_cols = list(set(ham_df.columns).intersection(set(isic_df.columns)))

ham_df = ham_df[common_cols]
isic_df = isic_df[common_cols]

# =========================
# MERGE HAM + ISIC
# =========================
combined_df = pd.concat([ham_df, isic_df], ignore_index=True)
combined_df = combined_df.drop_duplicates(subset="image_id")

# =========================
# SAVE
# =========================
output_path = "datasets/metadata_combined.csv"
combined_df.to_csv(output_path, index=False)

print("\nMerged successfully ✅")
print("Total combined samples:", len(combined_df))