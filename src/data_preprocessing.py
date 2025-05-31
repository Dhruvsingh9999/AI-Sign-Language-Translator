import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_dir=r"D:\Sign_Language_Translator\dataset\asl_alphabet_train", img_size=(64, 64)):
    X = []
    y = []
    class_names = sorted(os.listdir(data_dir))

    for idx, class_name in enumerate(class_names):
        folder_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(folder_path):
            continue
        print(f"Loading images from: {class_name}")
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            try:
                img = cv2.imread(file_path)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    X.append(img)
                    y.append(idx)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    X = np.array(X) / 255.0  # Normalize
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, class_names
