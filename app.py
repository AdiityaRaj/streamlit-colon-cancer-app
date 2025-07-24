import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from PIL import Image

# Constants
MODEL_PATH = "densenet.h5"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Clear previous uploads
def clear_upload_folder():
    for f in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

clear_upload_folder()

# Load the model
model = load_model(MODEL_PATH, compile=False)

# Define class names
class_names = [
    'dyed-lifted-polyps',
    'dyed-resection-margins',
    'esophagitis',
    'normal-cecum',
    'normal-pylorus',
    'normal-z-line',
    'polyps',
    'ulcerative-colitis'
]

# Page config
st.set_page_config(page_title="Colon Cancer Prediction", layout="centered")

# Title and instructions
st.markdown("<h1 style='text-align: center;'>🔬 Colon Cancer Type Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an endoscopic image to detect the cancer class and confidence score.</p>", unsafe_allow_html=True)

# Upload widget
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    clear_upload_folder()
    image = Image.open(uploaded_file).convert("RGB")

    # Show image centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    image.save(image_path)

    # Preprocess image
    img = load_img(image_path, target_size=(75, 100))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Spinner while predicting
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.spinner("🧠 AI is analyzing the image..."):
            preds = model.predict(img_array)
            predicted_class = class_names[np.argmax(preds)]
            confidence = round(np.max(preds) * 100, 2)

    # Show results
    st.markdown("<h3 style='text-align: center;'>Prediction Result</h3>", unsafe_allow_html=True)

    st.markdown(f"""
        <div style='text-align: center; padding: 10px; background-color: #ecf0f1; border-radius: 10px; margin-top: 10px;'>
            <p style='font-size: 22px; color: #27ae60;'><b>Class:</b> {predicted_class}</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style='text-align: center; padding: 10px; background-color: #ecf0f1; border-radius: 10px; margin-top: 10px;'>
            <p style='font-size: 20px; color: #2c3e50;'><b>Confidence:</b> {confidence}%</p>
        </div>
    """, unsafe_allow_html=True)
