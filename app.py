import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Flood Detection",
    page_icon="🌊",
    layout="wide"
)


# =========================
# CONFIGURATION
# =========================

MODEL_PATH = "model/efficientnetv2_flood_model.keras"
IMG_SIZE = (224, 224)


# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# =========================
# PREPROCESS IMAGE
# =========================

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# =========================
# HEADER
# =========================

st.title("🌊 Flood Detection")
st.write(
    "Aplikasi klasifikasi citra untuk mengidentifikasi gambar "
    "Flood dan Non Flood menggunakan model EfficientNetV2B0."
)

st.divider()


# =========================
# MAIN LAYOUT
# =========================

left, right = st.columns(2, gap="large")


# =========================
# LEFT COLUMN
# =========================

with left:
    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Pilih gambar yang ingin diklasifikasikan",
        type=["jpg", "jpeg", "png"],
        help="Format yang didukung: JPG, JPEG, dan PNG."
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Image Preview",
            use_container_width=True
        )


# =========================
# RIGHT COLUMN
# =========================

with right:
    st.subheader("🔍 Prediction Result")

    if uploaded_file is None:
        st.info(
            "Upload gambar terlebih dahulu untuk memulai prediksi."
        )

    else:
        predict_button = st.button(
            "Predict Image",
            type="primary",
            use_container_width=True
        )

        if predict_button:
            try:
                with st.spinner("Menganalisis gambar..."):
                    model = load_model()

                    processed_image = preprocess_image(image)

                    prediction = float(
                        model.predict(
                            processed_image,
                            verbose=0
                        )[0][0]
                    )

                # Class mapping:
                # 0 = Flood
                # 1 = Non Flood
                if prediction >= 0.5:
                    label = "Non Flood"
                    confidence = prediction * 100
                else:
                    label = "Flood"
                    confidence = (1 - prediction) * 100

                confidence_text = (
                    "> 99.99%"
                    if confidence >= 99.995
                    else f"{confidence:.2f}%"
                )

                st.success("Prediction completed successfully.")

                st.metric(
                    label="Prediction",
                    value=label
                )

                st.metric(
                    label="Confidence Score",
                    value=confidence_text
                )

                st.progress(
                    min(int(confidence), 100)
                )

                if label == "Flood":
                    st.warning(
                        "Gambar diklasifikasikan sebagai Flood."
                    )
                else:
                    st.info(
                        "Gambar diklasifikasikan sebagai Non Flood."
                    )

            except Exception as error:
                st.error(
                    f"Terjadi kesalahan saat melakukan prediksi: {error}"
                )


# =========================
# MODEL INFORMATION
# =========================

st.divider()

with st.expander("ℹ️ Model Information"):
    st.write("**Model Architecture:** EfficientNetV2B0")
    st.write("**Input Size:** 224 × 224 pixels")
    st.write("**Classification:** Binary Classification")
    st.write("**Classes:** Flood dan Non Flood")


# =========================
# DISCLAIMER
# =========================

st.caption(
    "Catatan: Hasil prediksi merupakan keluaran model machine learning "
    "dan dapat mengalami kesalahan klasifikasi pada kondisi tertentu."
)