import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Sahifa sozlamalari
st.set_page_config(page_title="Uni AI Detector", layout="centered")
st.title("🇺🇿 Universitet AI Detektori")
st.write("Kamerani yoqing va ob'ektni ko'rsating")

# Modelni yuklash (Keshlanadi, har safar yuklamaslik uchun)
@st.cache_resource
def load_model():
    model = YOLO('yolov8n.pt')
    # O'zbekcha nomlar lug'ati
    uni_names = {
        0: "Inson (Talaba)", 24: "Ryukzak", 26: "Sumka", 
        39: "Suv idishi", 41: "Kofe stakani", 63: "Noutbuk", 
        67: "Telefon", 73: "Kitob", 74: "Soat"
    }
    for id, name in uni_names.items():
        if id in model.model.names:
            model.model.names[id] = name
    return model

model = load_model()

# Kamera inputi
img_file_buffer = st.camera_input("Kameraga ko'rsating")

if img_file_buffer is not None:
    # Rasmni formatlash
    image = Image.open(img_file_buffer)
    img_array = np.array(image)
    
    # Bashorat qilish
    results = model(img_array)
    
    # Natijani chizish
    res_plotted = results[0].plot()
    
    # Ekranga chiqarish
    st.image(res_plotted, caption="AI natijasi", use_column_width=True)
