import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_pipeline import build_multimodal_dataset

# ===============================
# PATHS
# ===============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# 🚨 CHANGED: Now pointing to your newly trained V3 Hero Model
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "resnet50_V3_training.h5")

# ===============================
# LOAD MODEL & DATA
# ===============================
print("Loading Ultimate ResNet50 (V3) model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Loading Validation Data...")
val_ds, class_names, META_DIM, _ = build_multimodal_dataset("val", batch_size=16)

# ===============================
# PREDICT
# ===============================
print("Running predictions... (This might take a minute)")
y_true = []
y_pred = []

# Iterate through the validation dataset to guarantee labels match exactly
for (img_batch, meta_batch), label_batch in val_ds:
    preds = model.predict_on_batch([img_batch, meta_batch])
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(label_batch.numpy(), axis=1))

# ===============================
# METRICS & CONFUSION MATRIX
# ===============================
print("\n" + "="*55)
print("📊 CLASSIFICATION REPORT (ResNet50 V3 - Deep Fine-Tuned)")
print("="*55)
print(classification_report(y_true, y_pred, target_names=class_names))

# Create and save Confusion Matrix Heatmap
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[c.split(' ')[0] for c in class_names], 
            yticklabels=[c.split(' ')[0] for c in class_names])

plt.title('ResNet50 V3 Confusion Matrix (Validation Set)')
plt.ylabel('Actual True Disease')
plt.xlabel('AI Predicted Disease')

# Save the image to your main folder
cm_path = os.path.join(PROJECT_ROOT, "resnet50_V3_confusion_matrix.png")
plt.savefig(cm_path, bbox_inches="tight")
print(f"\n✅ Final Confusion Matrix image saved to: {cm_path}")