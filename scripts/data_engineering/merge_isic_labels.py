import pandas as pd

# =========================
# PATH TO ISIC TRAIN METADATA
# =========================
meta_path = "datasets/isic_raw/isic2018_task3_train/metadata.csv"

# Load metadata
df = pd.read_csv(meta_path)

# =========================
# LABEL MAPPING FUNCTION
# =========================
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

# =========================
# APPLY MAPPING
# IMPORTANT: using diagnosis_3 column
# =========================
df["dx"] = df["diagnosis_3"].apply(map_label)

# =========================
# SAVE NEW MAPPED FILE
# =========================
output_path = "datasets/isic_raw/isic2018_task3_train/metadata_mapped.csv"
df.to_csv(output_path, index=False)

print("Mapping done ✅")
print("Saved to:", output_path)