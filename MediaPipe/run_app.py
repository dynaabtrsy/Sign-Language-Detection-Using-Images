import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

LABELS = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','space']

try:
    mlp_model = tf.keras.models.load_model("asl_mediapipe_mlp_model.h5")
    print("Pretrained model loaded successfully!")
except Exception as e:
    print(f"Error loading model. Make sure 'asl_mediapipe_mlp_model.h5' is in this directory. Details: {e}")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("Starting webcam... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            input_data = np.array(landmarks).flatten().reshape(1, -1)
            prediction = mlp_model.predict(input_data, verbose=0)
            class_id = np.argmax(prediction)
            confidence = np.max(prediction)

            label = LABELS[class_id] if class_id < len(LABELS) else str(class_id)
            text = f"{label} ({confidence*100:.1f}%)"
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Recognition System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
