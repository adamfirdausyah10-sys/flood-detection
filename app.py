import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="Flood Detection",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Flood Detection")
st.caption("Flood Image Classification using EfficientNetV2B0")

MODEL_PATH = "model/efficientnetv2_flood_model.keras"
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

left, right = st.columns(2)

with left:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a flood/non-flood image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Preview Image", use_container_width=True)

with right:
    st.subheader("Prediction Result")

    if uploaded_file:
        if st.button("Predict"):
            model = load_model()
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)[0][0]
            st.caption(f"Raw model output: {prediction:.6f}")

            if prediction >= 0.5:
                label = "Non Flood"
                confidence = prediction * 100
            else:
                label = "Flood"
                confidence = (1 - prediction) * 100

            st.success(f"Prediction: {label}")
            confidence_text = "> 99.99%" if confidence >= 99.995 else f"{confidence:.2f}%"
            st.metric("Confidence", confidence_text)
            st.progress(int(confidence))
    else:
        st.info("Upload an image first to start prediction.")