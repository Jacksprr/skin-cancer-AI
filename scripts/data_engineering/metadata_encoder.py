# project/metadata_encoder.py

import pandas as pd
import numpy as np
import os

# ===== PATH TO TRAINING METADATA =====
META_CSV = "datasets/metadata_processed.csv"

# ===== LOAD METADATA TEMPLATE =====
df = pd.read_csv(META_CSV)

# Columns NOT used as model input
IGNORE_COLS = ["lesion_id", "image_id", "dx", "dx_type"]

# IMPORTANT: keep feature order EXACTLY same as training
FEATURE_COLUMNS = [c for c in df.columns if c not in IGNORE_COLS]

print("Metadata feature columns:")
print(FEATURE_COLUMNS)


# ===== LOAD NORMALIZATION STATS (IF AVAILABLE) =====
MEAN_PATH = "models/meta_mean.npy"
STD_PATH = "models/meta_std.npy"

if os.path.exists(MEAN_PATH) and os.path.exists(STD_PATH):
    META_MEAN = np.load(MEAN_PATH)
    META_STD = np.load(STD_PATH)
    print("Loaded metadata normalization stats.")
else:
    META_MEAN = None
    META_STD = None
    print("WARNING: Normalization stats not found. Using raw metadata.")


# ===== BUILD METADATA VECTOR =====
def build_metadata(age=None, sex=None, localization=None):

    # initialize empty vector
    meta = {col: 0.0 for col in FEATURE_COLUMNS}

    # AGE
    if age is not None and "age" in meta:
        meta["age"] = float(age)

    # SEX ONE-HOT
    if sex is not None:
        sex_col = f"sex_{sex.lower()}"
        if sex_col in meta:
            meta[sex_col] = 1.0

    # LOCALIZATION ONE-HOT
    if localization is not None:
        loc_col = f"loc_{localization.lower().replace(' ', '_')}"
        if loc_col in meta:
            meta[loc_col] = 1.0

    # convert to numpy in SAME order
    meta_array = np.array([meta[col] for col in FEATURE_COLUMNS], dtype=np.float32)

    # 🔥 Apply normalization ONLY if stats exist
    if META_MEAN is not None and META_STD is not None:
        meta_array = (meta_array - META_MEAN) / META_STD

    # add batch dimension
    meta_array = np.expand_dims(meta_array, axis=0)

    return meta_array
