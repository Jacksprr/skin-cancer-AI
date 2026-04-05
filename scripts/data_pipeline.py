import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf

# ===============================
# PATHS
# ===============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "combined_split")
METADATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "HAM10000_metadata.csv")

# 🚀 UPGRADE: 300x300 High Resolution
IMG_SIZE = (300, 300)

# ===============================
# LOAD + PREPROCESS METADATA 
# ===============================
def load_metadata():
    print("Processing Metadata...")
    df = pd.read_csv(METADATA_PATH)

    df["age"] = df["age"].fillna(df["age"].median())
    df["sex"] = df["sex"].fillna("unknown")
    df["localization"] = df["localization"].fillna("unknown")

    df["age"] = df["age"] / 100.0

    sex_encoded = pd.get_dummies(df["sex"], prefix="sex")
    loc_encoded = pd.get_dummies(df["localization"], prefix="loc")

    metadata_features = pd.concat([df["age"], sex_encoded, loc_encoded], axis=1)

    metadata_dict = dict(
        zip(df["image_id"], metadata_features.values.astype(np.float32))
    )

    return metadata_dict, metadata_features.shape[1]

META_DICT, META_DIM = load_metadata()
FALLBACK_META = np.zeros(META_DIM, dtype=np.float32)

def clean_image_id(filename):
    base = os.path.splitext(filename)[0]
    match = re.search(r'(ISIC_\d+|HAM_\d+)', base, re.IGNORECASE)
    if match:
        return match.group(1)
    return base 

# ===============================
# LOAD IMAGE PATHS + LABELS + META
# ===============================
def load_data_lists(split):
    split_path = os.path.join(DATA_DIR, split)

    class_names = sorted([
        d for d in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, d))
    ])

    filepaths = []
    labels = []
    metadata_vectors = [] 

    for idx, cls in enumerate(class_names):
        cls_path = os.path.join(split_path, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for img_name in images:
            filepaths.append(os.path.join(cls_path, img_name))
            labels.append(idx)
            
            clean_id = clean_image_id(img_name)
            meta_vec = META_DICT.get(clean_id, FALLBACK_META)
            metadata_vectors.append(meta_vec)

    return filepaths, labels, metadata_vectors, class_names

# ===============================
# IMAGE AUGMENTATION
# ===============================
def augment_image(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    return img

# ===============================
# MULTIMODAL DATASET BUILDER
# ===============================
def build_multimodal_dataset(split, batch_size):
    filepaths, labels, metadata_vectors, class_names = load_data_lists(split)

    NUM_CLASSES = len(class_names)
    total_images = len(filepaths)

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, metadata_vectors, labels))

    if split == "train":
        dataset = dataset.shuffle(total_images)
        dataset = dataset.map(
            lambda path, meta, label: (
                (
                    augment_image(tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(path), channels=3), IMG_SIZE)),
                    meta
                ),
                tf.one_hot(label, NUM_CLASSES)
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        dataset = dataset.map(
            lambda path, meta, label: (
                (
                    tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(path), channels=3), IMG_SIZE),
                    meta
                ),
                tf.one_hot(label, NUM_CLASSES)
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, class_names, META_DIM, total_images