import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('hand_gesture_model.h5')

# Label map for gestures
label_map = {
    0: 'Palm',
    1: 'L',
    2: 'Fist',
    3: 'Fist_moved',
    4: 'Thumb',
    5: 'Index',
    6: 'OK',
    7: 'Palm_moved',
    8: 'C',
    9: 'Down'
}

# Preprocess function
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 64, 64, 1)
    return reshaped


# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to access camera.")
    exit()

print("📷 Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Flip frame for mirror view
    frame = cv2.flip(frame, 1)

    # Convert to grayscale for display
    gray_display = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(gray_display, cv2.COLOR_GRAY2BGR)  # Keep 3 channels for OpenCV drawing

    # ROI coordinates
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    # Predict gesture
    input_frame = preprocess(roi)
    predictions = model.predict(input_frame)
    gesture_id = np.argmax(predictions)
    gesture_name = label_map.get(gesture_id, "Unknown")

    # Draw ROI and label
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f'Gesture: {gesture_name}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition (Grayscale View)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("Hand Gesture Recognition (Grayscale View)", cv2.WND_PROP_VISIBLE) < 1:
        break
cap.release()
cv2.destroyAllWindows()
