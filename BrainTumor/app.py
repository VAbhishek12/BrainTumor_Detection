import os
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import requests
import io

# Google Drive model and label encoder links (use your own)
MODEL_URL = "https://drive.google.com/file/d/1njXHe--omta3T9OtFYAEJCghOsW4UBHO/view?usp=drive_link"
LABEL_ENCODER_URL = "https://drive.google.com/file/d/1WJBe8ePRoWJ9eYaaKVTe5l5DQIqUbytd/view?usp=drive_link"

# File paths
MODEL_PATH = "brain_tumor_model.h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# Function to download files from Google Drive
def download_file(url, output_path):
    if not os.path.exists(output_path):
        with st.spinner(f"📥 Downloading {output_path}..."):
            response = requests.get(url, stream=True)
            with open(output_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        st.success(f"✅ {output_path} downloaded!")

# Download model & encoder if not present
download_file(MODEL_URL, MODEL_PATH)
download_file(LABEL_ENCODER_URL, LABEL_ENCODER_PATH)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Load the label encoder
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

        # Convert to grayscale if needed
        img = img.convert("L")

        # Resize to match model input (200x200)
        img = img.resize((200, 200))

        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0

        # Reshape to (1, 200, 200, 1) -> Model's expected input shape
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension

        return img_array, img  # Return both preprocessed array and original image
    except Exception as e:
        st.error(f"⚠️ Error in processing image: {str(e)}")
        return None, None

# Streamlit UI
st.title("🧠 Brain Tumor Detection")
st.write("Upload an image or provide an image URL for prediction.")

# Option to upload image
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "png", "jpeg"])

# Option to enter an image URL
image_url = st.text_input("🌐 Or enter an image URL:")

img_array = None
display_image = None

if uploaded_file is not None:
    # Process uploaded image
    img_array, display_image = load_and_preprocess_image(uploaded_file)

elif image_url:
    img_array, display_image = load_and_preprocess_image(image_url)

# Display Image
if display_image is not None:
    st.image(display_image, caption="🖼️ Processed Image", use_column_width=True)

# Make Prediction
if img_array is not None:
    with st.spinner("🔍 Analyzing Image..."):
        prediction = model.predict(img_array)
    
    # Get classification result
    class_index = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100

    # Get class label from encoder
    class_label = label_encoder.inverse_transform([class_index])[0]

    # Display result
    st.write(f"### Prediction: **{class_label}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
