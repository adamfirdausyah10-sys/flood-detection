import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# =========================
# KONFIGURASI
# =========================

parser = argparse.ArgumentParser(
    description="Evaluate Flood Detection model"
)

parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Path menuju folder dataset"
)

args = parser.parse_args()

DATASET_PATH = args.dataset
MODEL_PATH = "model/efficientnetv2_flood_model.keras"
OUTPUT_DIR = "outputs"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

CLASS_NAMES = [
    "Flood Images",
    "Non Flood Images"
]

DISPLAY_NAMES = [
    "Flood",
    "Non Flood"
]

# =========================
# SIAPKAN FOLDER OUTPUT
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD VALIDATION DATASET
# =========================

print("Loading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# =========================
# LOAD MODEL
# =========================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# =========================
# PREDIKSI
# =========================

print("Evaluating model...")

y_true = []
y_pred = []

for images, labels in val_ds:

    probabilities = model.predict(
        images,
        verbose=0
    ).ravel()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    y_true.extend(
        labels.numpy()
        .astype(int)
        .ravel()
    )

    y_pred.extend(predictions)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# =========================
# CLASSIFICATION REPORT
# =========================

report = classification_report(
    y_true,
    y_pred,
    target_names=DISPLAY_NAMES,
    digits=4
)

print("\n=== CLASSIFICATION REPORT ===")
print(report)

report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n=== CONFUSION MATRIX ===")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=DISPLAY_NAMES
)

disp.plot()

plt.title(
    "Confusion Matrix - Flood Detection"
)

plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =========================
# SELESAI
# =========================

print("\nEvaluation completed!")
print(f"Classification report saved to: {report_path}")
print(f"Confusion matrix saved to: {cm_path}")