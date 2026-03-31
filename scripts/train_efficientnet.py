import os
import math
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from data_pipeline import build_multimodal_dataset, load_data_lists

# ===============================
# CPU Optimization
# ===============================
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

# 🚀 LOWERED BATCH SIZE to prevent memory crash with larger 300x300 images
BATCH_SIZE = 10   
EPOCHS_STAGE1 = 20 
EPOCHS_STAGE2 = 30 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# LOAD DATASETS & CLASS WEIGHTS
# ===============================
print("Loading multimodal datasets (High-Res Mode)...")

train_ds, class_names, META_DIM, train_size = build_multimodal_dataset("train", BATCH_SIZE)
val_ds, _, _, _ = build_multimodal_dataset("val", BATCH_SIZE)

NUM_CLASSES = len(class_names)
_, train_labels, _, _ = load_data_lists("train")

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = dict(enumerate(class_weights))
dampened_weight_dict = {i: math.sqrt(weight) for i, weight in class_weight_dict.items()}

# ===============================
# BUILD MULTIMODAL MODEL (300x300)
# ===============================
def build_multimodal_model():
    # 🚀 UPGRADE: 300x300 Input Shape
    img_inputs = tf.keras.Input(shape=(300, 300, 3), name="image_input")
    
    # We stick with B0, but it will scale up to the new pixel density
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, 
        weights="imagenet", 
        input_tensor=img_inputs
    )
    base_model.trainable = False 
    
    x = tf.keras.applications.efficientnet.preprocess_input(img_inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    img_out = tf.keras.layers.Dropout(0.5)(x) 

    meta_inputs = tf.keras.Input(shape=(META_DIM,), name="meta_input")
    y = tf.keras.layers.Dense(64, activation="relu")(meta_inputs)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Dropout(0.3)(y)
    meta_out = tf.keras.layers.Dense(32, activation="relu")(y)

    combined = tf.keras.layers.Concatenate()([img_out, meta_out])
    
    z = tf.keras.layers.Dense(128, activation="relu")(combined)
    z = tf.keras.layers.Dropout(0.3)(z)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="final_output")(z)

    model = tf.keras.Model(inputs=[img_inputs, meta_inputs], outputs=outputs)
    return model, base_model

# ===============================
# CALLBACKS 
# ===============================
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "best_multimodal.keras"),
    monitor="val_accuracy", 
    save_best_only=True,
    mode="max",
    verbose=1
)

lr_scheduler_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-7, mode="max", verbose=1
)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=8, restore_best_weights=True, mode="max", verbose=1
)

callbacks_list = [checkpoint_cb, lr_scheduler_cb, early_stopping_cb]

# ===============================
# TRAINING
# ===============================
model, base_model = build_multimodal_model()

# 🚀 UPGRADE: Categorical Focal Crossentropy
focal_loss = tf.keras.losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0)

model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-4),
    loss=focal_loss, 
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

print("\n🚀 Stage 1: Training Top Layers & Metadata Branch (Focal Loss Engine)\n")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weight_dict, 
    callbacks=callbacks_list
)

print("\n🚀 Stage 2: Fine-tuning Base Model & Calibrating (Focal Loss Engine)\n")

base_model.trainable = True
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5), 
    loss=focal_loss, 
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    class_weight=dampened_weight_dict, 
    callbacks=callbacks_list
)

model.save(os.path.join(MODEL_DIR, "final_multimodal.keras"))
print("✅ High-Res Multimodal Training complete!")