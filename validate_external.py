import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
)


# =========================
# CONFIGURATION
# =========================

MODEL_PATH = Path(
    "model/efficientnetv2_flood_model.keras"
)

SAMPLE_DIR = Path(
    "sample_images"
)

OUTPUT_DIR = Path(
    "outputs"
)

IMG_SIZE = (224, 224)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


# Mapping model:
# 0 = Flood
# 1 = Non Flood

CLASS_CONFIG = {
    "flood": {
        "expected_id": 0,
        "display_name": "Flood",
    },
    "non-flood": {
        "expected_id": 1,
        "display_name": "Non Flood",
    },
}


# =========================
# PREPROCESS IMAGE
# =========================

def preprocess_image(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMG_SIZE)

        img_array = np.array(image)
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

    return img_array


# =========================
# LOAD MODEL
# =========================

print("Loading model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


# =========================
# PREPARE OUTPUT
# =========================

OUTPUT_DIR.mkdir(
    exist_ok=True
)

results = []

y_true = []
y_pred = []


# =========================
# VALIDATE EXTERNAL IMAGES
# =========================

print("Starting external validation...\n")

for folder_name, config in CLASS_CONFIG.items():

    folder_path = SAMPLE_DIR / folder_name

    if not folder_path.exists():
        print(
            f"WARNING: Folder tidak ditemukan: "
            f"{folder_path}"
        )
        continue

    image_paths = sorted(
        path
        for path in folder_path.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    )

    print(
        f"Folder '{folder_name}': "
        f"{len(image_paths)} image(s)"
    )

    for image_path in image_paths:

        expected_id = config["expected_id"]
        expected_label = config["display_name"]

        try:
            processed_image = preprocess_image(
                image_path
            )

            raw_output = float(
                model.predict(
                    processed_image,
                    verbose=0
                )[0][0]
            )

            # Mapping:
            # output >= 0.5 -> Non Flood
            # output < 0.5  -> Flood

            if raw_output >= 0.5:
                predicted_id = 1
                predicted_label = "Non Flood"
                confidence = raw_output * 100
            else:
                predicted_id = 0
                predicted_label = "Flood"
                confidence = (
                    1 - raw_output
                ) * 100

            is_correct = (
                expected_id == predicted_id
            )

            y_true.append(expected_id)
            y_pred.append(predicted_id)

            results.append({
                "filename": image_path.name,
                "expected": expected_label,
                "predicted": predicted_label,
                "confidence_percent": round(
                    confidence,
                    4
                ),
                "raw_output": round(
                    raw_output,
                    6
                ),
                "correct": is_correct,
                "error": "",
            })

            status = (
                "CORRECT"
                if is_correct
                else "WRONG"
            )

            print(
                f"[{status}] "
                f"{image_path.name} | "
                f"Expected: {expected_label} | "
                f"Predicted: {predicted_label} | "
                f"Confidence: {confidence:.2f}%"
            )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError
        ) as error:

            results.append({
                "filename": image_path.name,
                "expected": expected_label,
                "predicted": "",
                "confidence_percent": "",
                "raw_output": "",
                "correct": False,
                "error": str(error),
            })

            print(
                f"[ERROR] "
                f"{image_path.name}: {error}"
            )


# =========================
# SUMMARY
# =========================

if not y_true:
    raise RuntimeError(
        "Tidak ada gambar valid yang berhasil diuji."
    )

total_images = len(y_true)

correct_predictions = sum(
    true == pred
    for true, pred in zip(
        y_true,
        y_pred
    )
)

accuracy = (
    correct_predictions
    / total_images
    * 100
)

print("\n=== EXTERNAL VALIDATION SUMMARY ===")

print(
    f"Total images : {total_images}"
)

print(
    f"Correct      : {correct_predictions}"
)

print(
    f"Wrong        : "
    f"{total_images - correct_predictions}"
)

print(
    f"Accuracy     : {accuracy:.2f}%"
)


# =========================
# SAVE CSV
# =========================

csv_path = (
    OUTPUT_DIR
    / "external_validation_results.csv"
)

fieldnames = [
    "filename",
    "expected",
    "predicted",
    "confidence_percent",
    "raw_output",
    "correct",
    "error",
]

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


# =========================
# EXTERNAL CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Flood",
        "Non Flood"
    ]
)

disp.plot()

plt.title(
    "External Validation Confusion Matrix"
)

plt.tight_layout()

cm_path = (
    OUTPUT_DIR
    / "external_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================
# FINISH
# =========================

print("\nExternal validation completed!")

print(
    f"CSV saved to: {csv_path}"
)

print(
    f"Confusion matrix saved to: {cm_path}"
)