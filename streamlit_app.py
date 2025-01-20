import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

st.title("Image Classification with Keras")

# Load the model and class labels
model = load_model("/model/keras_model.h5", compile=False)
class_names = open("/model/labels.txt", "r").readlines()

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Prepare the image for prediction
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Display the prediction results
    st.subheader("Prediction")
    st.write(f"Class: {class_name}")
    st.write(f"Confidence Score: {confidence_score:.2f}")