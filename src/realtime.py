import cv2
from tensorflow import keras
import numpy as np


# ----------------------------------
# Load The Detection Model
# ----------------------------------

model = keras.models.load_model(r"models\best_model.keras")
# Emotion labels
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


# ----------------------------------
# Load Haar Cascade
# ----------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open Webcam

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

while True:
    success, frame = cap.read()

    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Crop the face
        face = gray[y:y+h, x:x+w]

        # Resize to the CNN input size
        face = cv2.resize(face, (48, 48))

        # Show the cropped face
        cv2.imshow("Detected Face", face)

        face = face.astype("float32")

        face = np.expand_dims(face, axis=-1)

        face = np.expand_dims(face, axis=0)

        # -----------------------
        # Prediction
        # -----------------------
        prediction = model.predict(face, verbose=0)

        emotion_index = np.argmax(prediction)

        confidence = np.max(prediction)

        emotion = emotion_labels[emotion_index]

        # -----------------------
        # Display Emotion
        # -----------------------
        cv2.putText(
            frame,
            f"{emotion} ({confidence:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )




    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()