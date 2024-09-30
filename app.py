import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import segmentation_models as sm
from tensorflow.keras import backend as K

# Set up environment for Keras
os.environ["SM_FRAMEWORK"] = "tf.keras"

# Load the model with error handling
weights = [0.1666] * 6
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

model_path = "model/satellite_standard_unet_100epochs.hdf5"
custom_objects = {
    "dice_loss_plus_1focal_loss": total_loss,
    "jacard_coef": jacard_coef
}

# Attempt to load the model
try:
    model = load_model(model_path, custom_objects=custom_objects)
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()  # Stop the app if the model cannot be loaded

# Streamlit app
st.title("Satellite Image Segmentation")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and process the image
    test_img = Image.open(uploaded_file)
    desired_width, desired_height = 256, 256
    test_img = test_img.resize((desired_width, desired_height))
    test_img = np.array(test_img)

    # Prepare the image for the model
    test_img_input = np.expand_dims(test_img, axis=0)

    # Make the prediction
    prediction = model.predict(test_img_input)
    predicted_img = np.argmax(prediction, axis=3)[0, :, :]

    # Plotting the images
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    
    ax[0].imshow(test_img)
    ax[0].set_title('Testing Image')
    ax[0].axis('off')

    ax[1].imshow(predicted_img)
    ax[1].set_title('Prediction on Test Image')
    ax[1].axis('off')

    st.pyplot(fig)
