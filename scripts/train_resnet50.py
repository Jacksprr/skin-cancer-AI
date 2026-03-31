import os
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from data_pipeline import build_multimodal_dataset, load_data_lists

# ===============================
# CPU/GPU Optimization
# ===============================
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

BATCH_SIZE = 16   
EPOCHS_STAGE1 = 10  # Build the custom head
EPOCHS_STAGE2 = 30  # Deep Fine-Tuning (The secret to 90%+ accuracy)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# LOAD DATASETS & PERFECT WEIGHTS
# ===============================
print("Loading datasets for Ultimate ResNet50 Training...")

train_ds, class_names, META_DIM, train_size = build_multimodal_dataset("train", BATCH_SIZE)
val_ds, _, _, _ = build_multimodal_dataset("val", BATCH_SIZE)

NUM_CLASSES = len(class_names)
_, train_labels, _, _ = load_data_lists("train")

# Standard balanced weights: This stops the NV bias but keeps accuracy high
base_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = dict(enumerate(base_weights))
print(f"Balanced Class Weights Applied: {class_weight_dict}")

# ===============================
# BUILD MULTIMODAL ARCHITECTURE
# ===============================
def build_best_resnet50():
    # 1. Image Branch
    img_inputs = tf.keras.Input(shape=(300, 300, 3), name="image_input")
    x = tf.keras.applications.resnet50.preprocess_input(img_inputs)
    
    base_model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=x)
    base_model.trainable = False  # Freeze for Stage 1
    
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    img_out = tf.keras.layers.Dropout(0.4)(x) 

    # 2. Metadata Branch
    meta_inputs = tf.keras.Input(shape=(META_DIM,), name="meta_input")
    y = tf.keras.layers.Dense(64, activation="relu")(meta_inputs)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Dropout(0.3)(y)
    meta_out = tf.keras.layers.Dense(32, activation="relu")(y)

    # 3. Combined Output
    combined = tf.keras.layers.Concatenate()([img_out, meta_out])
    z = tf.keras.layers.Dense(256, activation="relu")(combined)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.Dropout(0.4)(z)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="final_output")(z)

    model = tf.keras.Model(inputs=[img_inputs, meta_inputs], outputs=outputs)
    return model, base_model

# ===============================
# TRAINING SETUP
# ===============================
model, base_model = build_best_resnet50()
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# 🚨 CHANGED: Save the absolute best version of the model under a NEW NAME
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "resnet50_V3_training.h5"), 
    monitor="val_accuracy", 
    save_best_only=True,
    mode="max",
    verbose=1
)

# Dynamically lower the learning rate if the model gets stuck
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
)

callbacks_list = [checkpoint_cb, lr_scheduler, early_stopping]

# ===============================
# STAGE 1: TRAIN THE HEAD
# ===============================
print("\n🚀 Stage 1: Training Custom Classification Head\n")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=loss_fn, 
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weight_dict, 
    callbacks=callbacks_list
)

# ===============================
# STAGE 2: DEEP FINE-TUNING 
# ===============================
print("\n🚀 Stage 2: Deep Fine-Tuning ResNet50 (Unlocking deep layers)\n")
# We unfreeze the entire base model, but keep BatchNormalization layers frozen 
# to prevent the weights from collapsing.
base_model.trainable = True
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Use a very small learning rate so we don't destroy the pre-trained features
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
    loss=loss_fn, 
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weight_dict, 
    callbacks=callbacks_list
)

print("\n✅ Ultimate ResNet50 Training complete! Model saved as resnet50_V3_training.h5")