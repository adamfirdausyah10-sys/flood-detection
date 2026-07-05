import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image


# =========================
# CONFIGURATION
# =========================

MODEL_PATH = Path(
    "model/efficientnetv2_flood_model.keras"
)

CSV_PATH = Path(
    "outputs/external_validation_results.csv"
)

SAMPLE_DIR = Path(
    "sample_images"
)

OUTPUT_DIR = Path(
    "outputs/gradcam"
)

IMG_SIZE = (224, 224)


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
# FIND TARGET FEATURE LAYER
# =========================

def find_last_4d_layer(keras_model):
    """
    Mencari layer terakhir pada model utama
    yang menghasilkan feature map 4 dimensi.

    Pada arsitektur project ini, biasanya
    layer tersebut adalah output feature map
    dari EfficientNetV2B0.
    """

    for layer in reversed(keras_model.layers):
        try:
            output_shape = layer.output.shape

            if len(output_shape) == 4:
                return layer

        except (AttributeError, TypeError):
            continue

    raise ValueError(
        "Tidak menemukan layer 4D "
        "untuk Grad-CAM."
    )


target_layer = find_last_4d_layer(model)

print(
    f"Grad-CAM target layer: "
    f"{target_layer.name}"
)


# =========================
# PREPROCESS IMAGE
# =========================

def preprocess_image(image_path):
    with Image.open(image_path) as image:
        original_image = image.convert("RGB")

    resized_image = original_image.resize(
        IMG_SIZE
    )

    img_array = np.array(
        resized_image
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return (
        original_image,
        img_array
    )


# =========================
# GRAD-CAM
# =========================

def make_gradcam_heatmap(
    img_array,
    keras_model,
    feature_layer,
    target_class_id
):
    """
    Membuat Grad-CAM untuk model dengan
    EfficientNetV2B0 sebagai nested model.

    Class mapping:
    0 = Flood
    1 = Non Flood
    """

    # Cari posisi nested feature extractor
    target_index = keras_model.layers.index(
        feature_layer
    )

    with tf.GradientTape() as tape:

        x = tf.convert_to_tensor(
            img_array,
            dtype=tf.float32
        )

        # =========================
        # FORWARD SEBELUM TARGET
        # =========================

        for layer in keras_model.layers[
            :target_index
        ]:

            # InputLayer tidak perlu dipanggil
            if isinstance(
                layer,
                tf.keras.layers.InputLayer
            ):
                continue

            x = layer(
                x,
                training=False
            )

        # =========================
        # FEATURE MAP TARGET
        # =========================

        feature_maps = feature_layer(
            x,
            training=False
        )

        tape.watch(
            feature_maps
        )

        # =========================
        # FORWARD SETELAH TARGET
        # =========================

        x = feature_maps

        for layer in keras_model.layers[
            target_index + 1:
        ]:

            x = layer(
                x,
                training=False
            )

        predictions = x

        # Sigmoid output:
        # P(Non Flood)
        non_flood_probability = (
            predictions[:, 0]
        )

        if target_class_id == 1:
            # Non Flood
            class_score = (
                non_flood_probability
            )
        else:
            # Flood
            class_score = (
                1.0
                - non_flood_probability
            )

    # =========================
    # GRADIENT
    # =========================

    gradients = tape.gradient(
        class_score,
        feature_maps
    )

    if gradients is None:
        raise RuntimeError(
            "Gradient tidak berhasil dihitung."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    feature_maps = feature_maps[0]

    heatmap = tf.reduce_sum(
        feature_maps
        * pooled_gradients,
        axis=-1
    )

    # ReLU
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalisasi 0-1
    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    return heatmap.numpy()


# =========================
# CREATE OVERLAY
# =========================

def create_overlay(
    original_image,
    heatmap,
    alpha=0.4
):
    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    heatmap_image = Image.fromarray(
        heatmap_uint8
    )

    heatmap_image = heatmap_image.resize(
        original_image.size
    )

    heatmap_array = (
        np.array(heatmap_image)
        / 255.0
    )

    # Color map Grad-CAM
    colored_heatmap = plt.get_cmap(
        "jet"
    )(heatmap_array)[..., :3]

    original_array = (
        np.array(original_image)
        .astype(np.float32)
        / 255.0
    )

    overlay = (
        (1 - alpha) * original_array
        + alpha * colored_heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    return (
        heatmap_array,
        colored_heatmap,
        overlay
    )


# =========================
# FIND WRONG PREDICTIONS
# =========================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"CSV tidak ditemukan: {CSV_PATH}"
    )

wrong_predictions = []

with CSV_PATH.open(
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        is_correct = (
            row["correct"]
            .strip()
            .lower()
            == "true"
        )

        if not is_correct and not row["error"]:
            wrong_predictions.append(row)


print(
    f"Wrong predictions found: "
    f"{len(wrong_predictions)}"
)


# =========================
# PREPARE OUTPUT
# =========================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================
# PROCESS WRONG IMAGES
# =========================

for index, row in enumerate(
    wrong_predictions,
    start=1
):

    filename = row["filename"]
    expected = row["expected"]
    predicted = row["predicted"]
    confidence = row["confidence_percent"]

    # Tentukan folder berdasarkan label asli
    if expected == "Flood":
        folder_name = "flood"
    else:
        folder_name = "non-flood"

    image_path = (
        SAMPLE_DIR
        / folder_name
        / filename
    )

    if not image_path.exists():
        print(
            f"[SKIP] Image tidak ditemukan: "
            f"{image_path}"
        )
        continue

    # Predicted class ID
    if predicted == "Non Flood":
        predicted_class_id = 1
    else:
        predicted_class_id = 0

    print(
        f"\nProcessing {index}: "
        f"{filename}"
    )

    print(
        f"Expected   : {expected}"
    )

    print(
        f"Predicted  : {predicted}"
    )

    print(
        f"Confidence : {confidence}%"
    )

    original_image, img_array = (
        preprocess_image(image_path)
    )

    heatmap = make_gradcam_heatmap(
        img_array=img_array,
        keras_model=model,
        feature_layer=target_layer,
        target_class_id=predicted_class_id
    )

    _, colored_heatmap, overlay = (
        create_overlay(
            original_image,
            heatmap
        )
    )

    # =========================
    # SAVE ANALYSIS FIGURE
    # =========================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    # Original
    axes[0].imshow(
        original_image
    )

    axes[0].set_title(
        "Original Image"
    )

    axes[0].axis(
        "off"
    )

    # Heatmap
    axes[1].imshow(
        colored_heatmap
    )

    axes[1].set_title(
        "Grad-CAM Heatmap"
    )

    axes[1].axis(
        "off"
    )

    # Overlay
    axes[2].imshow(
        overlay
    )

    axes[2].set_title(
        "Grad-CAM Overlay"
    )

    axes[2].axis(
        "off"
    )

    fig.suptitle(
        f"Expected: {expected} | "
        f"Predicted: {predicted} | "
        f"Confidence: {confidence}%",
        fontsize=12
    )

    plt.tight_layout()

    output_filename = (
        f"{index:02d}_"
        f"{Path(filename).stem}_gradcam.png"
    )

    output_path = (
        OUTPUT_DIR
        / output_filename
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved to: {output_path}"
    )


# =========================
# FINISH
# =========================

print(
    "\nGrad-CAM analysis completed!"
)

print(
    f"Results saved in: "
    f"{OUTPUT_DIR}"
)