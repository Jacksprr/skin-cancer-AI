import pandas as pd
import os

# =========================
# FUNCTION TO CLEAN ISIC METADATA
# =========================
def prepare_isic_split(split_name):
    print(f"\nProcessing {split_name}...")

    base_path = f"datasets/isic_raw/isic2018_task3_{split_name}"
    meta_path = os.path.join(base_path, "metadata.csv")

    df = pd.read_csv(meta_path)

    # ---------- LABEL MAPPING ----------
    def map_label(x):
        x = str(x).lower()

        if "melanoma" in x:
            return "mel"
        elif "basal cell" in x:
            return "bcc"
        elif "squamous" in x or "actinic" in x:
            return "akiec"
        elif "nevus" in x:
            return "nv"
        elif "keratosis" in x:
            return "bkl"
        elif "dermatofibroma" in x:
            return "df"
        elif "vascular" in x:
            return "vasc"
        else:
            return None

    # Use diagnosis_3 column
    df["dx"] = df["diagnosis_3"].apply(map_label)

    # ---------- CREATE CLEAN FORMAT ----------
    df_clean = pd.DataFrame()
    df_clean["image_id"] = df["isic_id"]
    df_clean["dx"] = df["dx"]
    df_clean["age"] = df["age_approx"]
    df_clean["sex"] = df["sex"]

    # FIX: correct localization column name
    if "anatom_site_general" in df.columns:
        df_clean["localization"] = df["anatom_site_general"]
    else:
        df_clean["localization"] = None

    # Remove rows without valid class
    df_clean = df_clean[df_clean["dx"].notnull()]

    output_path = os.path.join(base_path, "isic_clean.csv")
    df_clean.to_csv(output_path, index=False)

    print(f"Saved clean file → {output_path}")
    print(f"Total usable samples: {len(df_clean)}")


# =========================
# RUN FOR BOTH TRAIN + VAL
# =========================
prepare_isic_split("train")
prepare_isic_split("val")

print("\nISIC 2018 train + val cleaned successfully ✅")