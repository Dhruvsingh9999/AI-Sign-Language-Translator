import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from text_to_speech import speak

# Page config
st.set_page_config(page_title="AI Sign Language Translator", layout="centered")

# ---------- CUSTOM CSS STYLING ----------
st.markdown("""
    <style>
        body {
            background-color: #cacaca;
        }
        .container {
            background-color: #ffa726;
            width: 500px;
            margin: auto;
            padding: 30px 40px;
            border-radius: 20px;
            box-shadow: 5px 5px 20px rgba(0,0,0,0.2);
        }
        .title {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            color: black;
        }
        .subtitle {
            text-align: center;
            color: black;
            font-size: 18px;
            margin-bottom: 30px;
        }
    </style>
    <div class="container">
        <div class="title">🤟 AI Sign Language Translator</div>
        <div class="subtitle">Upload an image or use webcam to detect ASL signs</div>
    """, unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
model = load_model(r"D:\Sign_Language_Translator\model\sign_model.h5")
with open(r"D:\Sign_Language_Translator\model\class_names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# ---------- FILE UPLOAD SECTION ----------
uploaded_img = st.file_uploader("📂 Upload Hand Sign Image", type=["jpg", "png"])

if uploaded_img:
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img, caption="Uploaded Image", channels="RGB")

    resized = cv2.resize(img, (64, 64)) / 255.0
    pred = model.predict(np.expand_dims(resized, axis=0))
    pred_index = np.argmax(pred)
    
    if pred_index < len(classes):
        label = classes[pred_index]
        confidence = np.max(pred) * 100
        st.success(f"Predicted Sign: {label}")
        st.info(f"Confidence: {confidence:.2f}%")

        if st.button("🔊 Speak"):
            speak(label)
    else:
        st.error("Prediction index out of class range!")

# ---------- WEBCAM SECTION ----------
if "run_webcam" not in st.session_state:
    st.session_state.run_webcam = False

if st.checkbox("📷 Use Webcam"):
    if st.button("Start Camera"):
        st.session_state.run_webcam = True

    if st.button("Stop Camera"):
        st.session_state.run_webcam = False

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while st.session_state.run_webcam:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to read from webcam.")
            break

        # Define hand ROI
        h, w, _ = frame.shape
        x1, y1, x2, y2 = w // 2 - 100, h // 2 - 100, w // 2 + 100, h // 2 + 100
        roi = frame[y1:y2, x1:x2]

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(roi_rgb, (64, 64)) / 255.0
        pred = model.predict(np.expand_dims(resized, axis=0))
        pred_index = np.argmax(pred)
        confidence = np.max(pred) * 100

        if pred_index < len(classes):
            label = classes[pred_index]
        else:
            label = "Unknown"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({confidence:.2f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

# ---------- CLOSE STYLED DIV ----------
st.markdown("</div>", unsafe_allow_html=True)
