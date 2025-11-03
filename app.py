import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ------------------------------------------------------
# Load the trained model
# ------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("saved_model/best_model.h5")
    return model

model = load_model()

# CIFAR-10 class labels
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ------------------------------------------------------
# Streamlit app UI
# ------------------------------------------------------
st.set_page_config(page_title="Image Classification using CNN", layout="centered")
st.title("🧠 Image Classification using CNN")
st.write("Upload an image and let the trained CNN model predict its class (CIFAR-10).")

# ------------------------------------------------------
# File uploader
# ------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display result
    st.success(f"✅ **Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

else:
    st.info("👆 Please upload an image to start classification.")
