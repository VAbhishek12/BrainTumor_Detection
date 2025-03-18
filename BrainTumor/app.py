import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import io

# Load the trained model
MODEL_PATH = "brain_tumor_model.h5"  # Change this to your model's path
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess images
def load_and_preprocess_image(image_source):
    if isinstance(image_source, str):  # If URL is given
        response = requests.get(image_source)
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

    return img_array

# Streamlit UI
st.title("Brain Tumor Detection Model")
st.write("Upload an image or provide an image URL for prediction.")

# Option to upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Option to enter an image URL
image_url = st.text_input("Or enter an image URL:")

if uploaded_file is not None:
    # Process uploaded image
    img_array = load_and_preprocess_image(uploaded_file)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

elif image_url:
    try:
        img_array = load_and_preprocess_image(image_url)
        st.image(image_url, caption="Image from URL", use_column_width=True)
    except:
        st.error("Failed to load the image from the URL. Check the link and try again.")
        img_array = None

# Make Prediction
if (uploaded_file or image_url) and img_array is not None:
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction, axis=1)[0]  # Assuming classification
    confidence = np.max(prediction) * 100

    # Change class names based on your model
    class_labels = ["No Tumor", "Tumor Detected"]

    st.write(f"### Prediction: **{class_labels[class_index]}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

