import cv2
import numpy as np
import joblib
import mediapipe as mp
import os
import time  # ⏱️ For tracking time

# Load model and label encoder
model = joblib.load("asl_mediapipe_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define unwanted labels
excluded_labels = {"del", "space", "nothing"}

# Filter the usable ASL letters
all_labels = list(label_encoder.classes_)
letters = [label for label in all_labels if label not in excluded_labels]
current_idx = 0

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Folder with reference gesture images
asl_image_folder = "asl_images"

# Start webcam
cap = cv2.VideoCapture(0)
print(f"Sign the letter: {letters[current_idx]}")

# Start the timer
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    pred_class = ""
    feedback_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                prediction_encoded = model.predict([landmarks])[0]
                pred_class_raw = label_encoder.inverse_transform([prediction_encoded])[0]

                # Skip prediction if it's in excluded labels
                if pred_class_raw not in excluded_labels:
                    pred_class = pred_class_raw
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                feedback_text = f"Landmark mismatch: found {len(landmarks)}"
    else:
        feedback_text = "No hands detected."

    # Check prediction
    if pred_class == letters[current_idx]:
        print(f"✅ Correct! That was {letters[current_idx]}")
        current_idx += 1
        if current_idx >= len(letters):
            end_time = time.time()
            total_time = end_time - start_time
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)
            print("🎉 You completed all ASL letters!")
            print(f"⏱️ Total time taken: {total_time:.2f} seconds ({minutes} min {seconds} sec)")
            break
        print(f"👉 Next: Sign the letter {letters[current_idx]}")

    # Display gesture image from asl_images folder
    ref_img = None
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate_path = os.path.join(asl_image_folder, f"{letters[current_idx]}{ext}")
        if os.path.exists(candidate_path):
            ref_img = cv2.imread(candidate_path)
            if ref_img is not None:
                ref_img = cv2.resize(ref_img, (150, 150))  # Resize as needed
                frame[10:160, 10:160] = ref_img
            else:
                print(f"⚠️ Found {candidate_path} but couldn't load it.")
            break
    else:
        print(f"❌ No reference image found for letter {letters[current_idx]}")

    # Add on-screen text
    if pred_class:
        cv2.putText(frame, f"Predicted: {pred_class}", (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    if feedback_text:
        cv2.putText(frame, feedback_text, (10, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if current_idx < len(letters):
        cv2.putText(frame, f"Sign this letter: {letters[current_idx]}", (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("ASL Sign Predictor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Game exited.")
        break

cap.release()
cv2.destroyAllWindows()
