# AI-Sign-Language-Translator
# 🤟 AI-Powered Sign Language Translator

This project uses a Convolutional Neural Network (CNN) model trained on American Sign Language (ASL) images to translate hand gestures into real-time text using webcam input. A Streamlit web app is used to display the result with an interactive interface.

---

## 📁 Folder Structure

Sign_Language_Translator/
│

├── dataset/ # ASL image dataset (not included here)

│

├── models/

│ └── sign_model.h5 # Trained model

│

├── src/

│ ├── app.py # Streamlit web app

│ ├── train_model.py # Model training script

│ ├── data_preprocessing.py # Dataset loader

│ ├── predict_realtime.py # OpenCV live prediction

│ └── text_to_speech.py # Optional TTS functionality

│

├── requirements.txt

└── README.md


---

## ⚙️ Technologies Used

- Python 🐍
- TensorFlow / Keras 🧠
- OpenCV 🎥
- Streamlit 🌐
- NumPy & scikit-learn 📊

---

## 🚀 How to Run

### 🔧 1. Clone the repository

```bash
git clone https://github.com/yourusername/Sign_Language_Translator.git
cd Sign_Language_Translator
```

### 📦 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 🧠 3. Train the model (optional if sign_model.h5 is already available)

```bash
python src/train_model.py
```

### 🌐 4. Run Streamlit Web App

```bash
streamlit run src/app.py
```

---

## 🧠 Model Details

- **Input:** 64x64 RGB images  
- **Architecture:**  
  - 3 Convolutional layers + MaxPooling  
  - Dense(128) + Dropout  
  - Final layer: softmax over ASL classes  
- **Accuracy:** ~95% on test dataset

---

## 📷 Example Output

![Screenshot 2025-05-31 200030](https://github.com/user-attachments/assets/34855730-572e-45f8-91e2-32744019095a)

---

## 🙋‍♂️ Author  
**Dhruv Singh Somvanshi**
 
- 🔗 [LinkedIn](https://www.linkedin.com/in/dhruv-pratap-singh-459524284)

---

⭐ _Star this repository if you found it useful!_


