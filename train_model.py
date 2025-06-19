import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# === Setup MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# === Dataset paths ===
dataset_path = "ASLdataset"
train_dir = os.path.join(dataset_path, "asl_alphabet_train")
test_dir = os.path.join(dataset_path, "asl_alphabet_test")

# === Preprocessing Function ===
def extract_landmarks_from_folder(folder_path, label):
    data, labels, count = [], [], 0
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping unreadable image: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            landmark_list = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])
            data.append(landmark_list)
            labels.append(label)
            count += 1
            if count % 50 == 0:
                print(f"🖐️ {count} valid hand images processed for label '{label}'")
    return data, labels

# === Try to Load Existing Model ===
model_path = "asl_mediapipe_model.pkl"
encoder_path = "label_encoder.pkl"

if os.path.exists(model_path) and os.path.exists(encoder_path):
    print("✅ Found saved model and label encoder. Skipping training...")
    clf = joblib.load(model_path)
    le = joblib.load(encoder_path)
else:
    # === Load training data ===
    print("\n🚀 Starting training data preprocessing...")
    X, y = [], []
    for label in sorted(os.listdir(train_dir)):
        folder = os.path.join(train_dir, label)
        if not os.path.isdir(folder):
            continue
        print(f"\n⏳ Processing label: {label}")
        data, labels = extract_landmarks_from_folder(folder, label)
        X.extend(data)
        y.extend(labels)
        print(f"✅ Finished and saved progress for label: {label}")

    if not X:
        raise ValueError("🚫 No training data found. Check your training folder structure.")

    # === Encode labels and Train model ===
    print("\n🔐 Encoding labels and training model...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X, y_encoded)

    # === Save model and encoder ===
    joblib.dump(clf, model_path)
    joblib.dump(le, encoder_path)
    print("💾 Model and label encoder saved!")

# === Load and Evaluate on Test Set (if valid) ===
X_test, y_test = [], []
print("\n🚀 Starting testing data preprocessing...")
for label in sorted(os.listdir(test_dir)):
    folder = os.path.join(test_dir, label)
    if not os.path.isdir(folder):
        continue
    print(f"\n⏳ Processing label: {label}")
    data, labels = extract_landmarks_from_folder(folder, label)
    X_test.extend(data)
    y_test.extend(labels)

if not X_test:
    print("\n⚠️ No test data found or usable. Skipping evaluation step.")
else:
    y_test_encoded = le.transform(y_test)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test_encoded, y_pred)
    print(f"\n🎯 Test Accuracy: {acc * 100:.2f}%")

# === Optional: Predict a Single Test Image ===
def predict_single_image(img_path, model, encoder):
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Could not read image.")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        prediction = model.predict([landmarks])[0]
        label = encoder.inverse_transform([prediction])[0]
        print(f"🧠 Prediction for {os.path.basename(img_path)}: {label}")
    else:
        print(f"🙅 No hand detected in {os.path.basename(img_path)}")

# Example usage:
# predict_single_image("ASLdataset/asl_alphabet_test/X_test.jpg", clf, le)

# === Release resources ===
hands.close()
