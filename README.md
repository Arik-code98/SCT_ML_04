# SCT_ML_04

# ✋ Hand Gesture Recognition (Grayscale)

This project uses a trained deep learning model to recognize hand gestures in real-time using your webcam. It processes the hand region in grayscale and predicts the gesture using a Keras model trained on 64x64 pixel inputs.

---

## 📦 Features

- 📷 Real-time hand gesture recognition from webcam input.
- ⚙️ Utilizes a CNN model (`hand_gesture_model.h5`) trained on grayscale hand images.
- 🎛️ Lightweight grayscale preprocessing for fast inference.
- 🖼️ Displays predicted gesture on video with bounding box.
- 🧠 Works with 10 common gestures.

---

## 🧠 Recognized Gestures

| Label ID | Gesture       |
|----------|---------------|
| 0        | Palm          |
| 1        | L             |
| 2        | Fist          |
| 3        | Fist_moved    |
| 4        | Thumb         |
| 5        | Index         |
| 6        | OK            |
| 7        | Palm_moved    |
| 8        | C             |
| 9        | Down          |

---

## 🎥 Webcam Setup Instructions
To ensure accurate and consistent gesture detection:

- ✅ Face the webcam directly and keep your hand inside the green box (Region of Interest).

- 🧼 Use a clean and uncluttered background to avoid misclassification.

- 💡 Avoid harsh or overly bright lighting. A softly lit room works best.

- ✋ Use one hand at a time, and keep it steady within the ROI.

- 📏 Keep your hand at a consistent distance from the camera (~30 cm recommended).

- 🖐️ Ensure your gestures match the training data (frontal hand view, fingers clear).

---

## 📦 Installation

 **Clone the repository**
   ```bash
   git clone https://github.com/Arik-code98/SCT_ML_04.git
   cd SCT_ML_04