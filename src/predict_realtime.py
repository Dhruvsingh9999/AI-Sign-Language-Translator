import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from text_to_speech import speak
import os

IMG_SIZE = 64
model = load_model(r"D:\Sign_Language_Translator\model\sign_model.h5")
classes = sorted(os.listdir(r"D:\Sign_Language_Translator\dataset\asl_alphabet_train"))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    roi = frame[100:300, 100:300]
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img / 255.0, axis=0)

    prediction = model.predict(img)
    label = classes[np.argmax(prediction)]

    cv2.rectangle(frame, (100, 100), (300, 300), (0,255,0), 2)
    cv2.putText(frame, label, (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imshow("Sign Detection", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        speak(label)  # speak the prediction
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
