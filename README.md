# 🎭 Real-Time Facial Expression Recognition

A deep learning project that performs **real-time facial emotion recognition** using a **Convolutional Neural Network (CNN) built from scratch** and trained on the **FER2013** dataset.

---

## 🚀 Features

- Custom CNN built from scratch using TensorFlow/Keras
- Trained on the FER2013 dataset (35,887 grayscale images)
- Data augmentation and class-weighted training
- Real-time emotion prediction from webcam
- Face detection using OpenCV Haar Cascade
- Flask web application (In Progress)

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn

---

## 📂 Project Structure

```
├── data/
├── models/
├── notebooks/
├── src/
├── requirements.txt
└── README.md
```

---

## 🧠 Model Pipeline

```
Webcam
    ↓
Face Detection
    ↓
Crop Face
    ↓
Resize (48×48)
    ↓
CNN
    ↓
Emotion Prediction
```

---

## 📊 Current Performance

| Metric | Value |
|---------|------:|
| Validation Accuracy | 52.76% |
| Classes | 7 |
| Dataset | FER2013 |

---

## 🚀 Future Improvements

- Improve recognition of Fear and Angry emotions
- Experiment with deeper CNN architectures
- Replace Haar Cascade with MediaPipe
- Deploy using Flask

---
