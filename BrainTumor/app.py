import os
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import requests
import io
import gdown  # ✅ Add this for downloading model files

# Define correct Google Drive direct download links
MODEL_URL = "https://drive.google.com/uc?id=1njXHe--omta3T9OtFYAEJCghOsW4UBHO"
LABEL_ENCODER_URL = "https://drive.google.com/uc?id=1WJBe8ePRoWJ9eYaaKVTe5l5DQIqUbytd"

# Define local paths
MODEL_PATH = "brain_tumor_model.h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# Function to download files if they don't exist
def download_file(url, output_path):
    if not os.path.exists(output_path):
        st.warning(f"⏳ Downloading {output_path}... Please wait.")
        gdown.download(url, output_path, quiet=False)

# Download and Load Model
download_file(MODEL_URL, MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

# Download and Load Label Encoder
download_file(LABEL_ENCODER_URL, LABEL_ENCODER_PATH)
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# Function to preprocess images
def load_and_preprocess_image(image_source):
    try:
        if isinstance(image_source, str):  # If URL is given
            response = requests.get(image_source, timeout=5)
            if response.status_code != 200:
                raise ValueError("Invalid URL or failed to download image.")
            img = Image.open(io.BytesIO(response.content))
        else:  # If file is uploaded
            img = Image.open(image_source)

        img = img.convert("L")  # Convert to grayscale
        img = img.resize((200, 200))  # Resize
        img_array = np.array(img) / 255.0  # Normalize

        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension

        return img_array, img
    except Exception as e:
        st.error(f"⚠️ Error in processing image: {str(e)}")
        return None, None

# Streamlit UI
st.title("🧠 Brain Tumor Detection")
st.write("Upload an image or provide an image URL for prediction.")

uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "png", "jpeg"])
image_url = st.text_input("🌐 Or enter an image URL:")

img_array = None
display_image = None

if uploaded_file is not None:
    img_array, display_image = load_and_preprocess_image(uploaded_file)
elif image_url:
    img_array, display_image = load_and_preprocess_image(image_url)

if display_image is not None:
    st.image(display_image, caption="🖼️ Processed Image", use_column_width=True)

if img_array is not None:
    with st.spinner("🔍 Analyzing Image..."):
        prediction = model.predict(img_array)
    
    class_index = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100

    class_label = label_encoder.inverse_transform([class_index])[0]

    st.write(f"### Prediction: **{class_label}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
