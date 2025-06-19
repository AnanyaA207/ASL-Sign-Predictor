# 🤟 ASL Sign Predictor (MediaPipe + Machine Learning)

Real-time American Sign Language (ASL) letter recognition game using your webcam, powered by MediaPipe hand tracking and a machine learning model. Learn and practice ASL one sign at a time with visual references!

---

## 📸 Demo



---

## 🧠 How It Works

This project uses **MediaPipe** to detect 3D hand landmarks and a **RandomForestClassifier** (trained on ASL alphabet gestures) to predict the signed letter.

- 👋 Detects hand via webcam
- 📍 Extracts 21 hand landmarks (x, y, z)
- 🧠 Classifies your sign using a trained `.pkl` model
- 📷 Shows a reference image for each letter so you can copy the pose

---

## 🏗️ Model Info

Model type: RandomForestClassifier (via scikit-learn)
Features: 21 3D hand landmarks → 63 values total (x, y, z for each point)
Training data: Preprocessed images from the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
Training method: MediaPipe's landmark extraction → fed into classical ML model

### 🔧 Requirements

Make sure you have these installed:

```bash
pip install opencv-python mediapipe scikit-learn numpy joblib
