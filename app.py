import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile


# Load model
model = YOLO('models/best.pt')

st.title("Traffic Sign Detection with YOLOv8 🚦")

# Sidebar options
option = st.sidebar.selectbox('Select input type:', ('Image', 'Video', 'Live Camera'))

if option == 'Image':
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        results = model.predict(source=image, save=False)
        annotated_img = results[0].plot()
        st.image(annotated_img, channels="BGR")

elif option == 'Video':
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame, save=False)
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR")
        cap.release()

elif option == 'Live Camera':
    st.write("Use your phone's IP webcam app like 'IP Webcam' and paste the URL here.")
    url = st.text_input("IP Camera URL (e.g., http://192.168.1.100:8080/video)")

    if url:
        cap = cv2.VideoCapture(url)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame, save=False)
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR")
