import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU for stability

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ------------------------------------------------------
# ✅ Set Streamlit page config (must be first Streamlit command)
# ------------------------------------------------------
st.set_page_config(page_title="🧠 CNN Image Classifier", layout="centered")

# ------------------------------------------------------
# Load the trained CNN model
# ------------------------------------------------------
@st.cache_resource
def load_cnn_model():
    try:
        model = tf.keras.models.load_model("saved_model/best_model.h5")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_cnn_model()

# CIFAR-10 class labels
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ------------------------------------------------------
# Streamlit App UI
# ------------------------------------------------------
st.title("🧠 Image Classification using CNN")
st.write("Upload an image and let the trained CNN model predict its class (CIFAR-10 dataset).")

# ------------------------------------------------------
# File uploader
# ------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if model is not None:
    if uploaded_file is not None:
        # Load and resize image
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((32, 32))
        st.image(image, caption="📸 Uploaded Image", use_container_width=True)

        # Preprocess for prediction
        img_array = np.array(image).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Display results
        st.success(f"✅ **Predicted Class:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")

    else:
        st.info("👆 Please upload an image to start classification.")
else:
    st.error("🚫 Model could not be loaded. Make sure 'saved_model/best_model.h5' exists.")
