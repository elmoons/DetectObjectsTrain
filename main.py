import streamlit as st
import requests
from PIL import Image
import io

# Интерфейс для загрузки файла
st.title("Drone Detection with YOLO11")
uploaded_file = st.file_uploader("Upload a video/image", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    # Отправка файла на FastAPI
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://127.0.0.1:8000/detect", files=files)

    # Отображение результата
    if response.status_code == 200:
        st.image(Image.open(uploaded_file))
        st.write("Detections:", response.json())
    else:
        st.write("Error in detection")


