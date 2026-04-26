import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

print("Starting ASL Detection System")

# ======================
# Load the trained model
# ======================
print("Loading model...")
model = load_model("asl_model.h5")
print("Model loaded successfully")

labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# ======================
# MediaPipe setup
# ======================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ======================
# Open webcam
# ======================
print("Opening webcam...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera not detected")
    exit()

print("Camera opened successfully")

# ======================
# Main Loop
# ======================
while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # Convert to RGB for mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            x_list = []
            y_list = []

            for lm in hand.landmark:

                x = int(lm.x * w)
                y = int(lm.y * h)

                x_list.append(x)
                y_list.append(y)

            xmin = max(min(x_list) - 20, 0)
            xmax = min(max(x_list) + 20, w)

            ymin = max(min(y_list) - 20, 0)
            ymax = min(max(y_list) + 20, h)

            roi = frame[ymin:ymax, xmin:xmax]

            if roi.size != 0:

                roi = cv2.resize(roi, (64,64))
                roi = roi / 255.0
                roi = roi.reshape(1,64,64,3)

                prediction = model.predict(roi, verbose=0)

                index = np.argmax(prediction)
                letter = labels[index]

                cv2.putText(
                    frame,
                    letter,
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0,255,0),
                    3
                )

    # Show frame
    cv2.imshow("ASL Gesture Recognition", frame)

    key = cv2.waitKey(1)

    if key == 27:   # ESC to exit
        break

# ======================
# Cleanup
# ======================
cap.release()
cv2.destroyAllWindows()

print("Program closed")