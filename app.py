import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import io
from PIL import Image

# Constants
MODEL_PATH = 'model/satellite_standard_unet_100epochs.hdf5'
IMG_HEIGHT = 256  # Set according to your model's expected input size
IMG_WIDTH = 256   # Set according to your model's expected input size

# Load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Color map for segmentation mask
def get_color_map(num_classes):
    color_map = plt.get_cmap("hsv", num_classes)
    return color_map

# Preprocess the image
def load_and_preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize if required
    return img_array

# Make prediction
def make_prediction(uploaded_file):
    processed_img = load_and_preprocess_image(uploaded_file)
    predictions = model.predict(processed_img)

    # Assuming multi-class segmentation, get the class with the highest score
    predicted_mask = np.argmax(predictions[0], axis=-1).astype(np.uint8)

    return predicted_mask

# Color the predicted mask
def colorize_mask(mask, num_classes):
    color_map = get_color_map(num_classes)
    color_mask = color_map(mask)
    return (color_mask[:, :, :3] * 255).astype(np.uint8)  # Convert to uint8

# Streamlit app layout
st.title("Semantic Segmentation using UNet")
st.header("Upload an image to predict the segmentation mask")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Predict button
    if st.button("Predict"):
        if model is not None:
            # Make the prediction
            predicted_mask = make_prediction(uploaded_file)
            
            # Colorize the predicted mask
            colored_mask = colorize_mask(predicted_mask, num_classes=6)  # Change 6 to your actual number of classes
            
            # Display results
            st.subheader("Predicted Segmentation Mask")
            st.image(colored_mask, caption="Predicted Mask", use_column_width=True)
        else:
            st.error("Model is not loaded. Please check the model path.")
