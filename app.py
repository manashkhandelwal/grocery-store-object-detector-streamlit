import streamlit as st
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Define folders
IMAGE_FOLDER = 'static/images'  # Pre-uploaded images
RESULT_FOLDER = 'results'

# Ensure results folder exists
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO('model/best.pt')

# Run inference and save the result
def process_image(model, image_path):
    results = model(image_path)
    results_img = results[0].plot()  # Render detection results
    result_img_path = os.path.join(RESULT_FOLDER, f'result_{os.path.basename(image_path)}')
    Image.fromarray(results_img).save(result_img_path)
    return result_img_path

# Streamlit app
def main():
    st.title("Grocery Store Object Detection")
    st.write("Detect objects in predefined grocery images using YOLOv8.")

    # Load the YOLO model
    model = load_model()

    # List all pre-uploaded images
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
    selected_image = st.selectbox("Select an image:", image_files)

    if selected_image:
        # Display the selected image
        image_path = os.path.join(IMAGE_FOLDER, selected_image)
        st.image(image_path, caption="Selected Image", use_column_width=True)

        if st.button("Detect Objects"):
            st.write("Processing the image...")

            # Run YOLO inference and display results
            result_img_path = process_image(model, image_path)

            # Display the processed image
            st.image(result_img_path, caption="Detected Objects", use_column_width=True)

if __name__ == "__main__":
    main()
