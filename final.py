import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import os

def process_image(image,path_image):
    # focntion avec les model
    model.predict(source=image, conf=0.25,save=True)
    #blur_image = Image.open(r'C:\Users\Carlt\Documents\phoenix\runs\detect\predict\image0.jpg').convert("RGB")
    blur_image = cv2.imread(r"runs\detect\predict\image0.jpg",1)
    gray_image = cv2.GaussianBlur(np.array(image), (15, 15), 0)

    return gray_image, blur_image

model = YOLO(r'best.pt')
st.title("Application de traitement d'images")
st.write("Veuillez fournir une image et l'application retournera deux autres images")

uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image originale", use_column_width=True)

    gray_image, blur_image = process_image(image,uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(gray_image, caption="Image en niveaux de gris")
    with col2:
        st.image(blur_image, caption="Image avec effet de flou")
