from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = FastAPI()

# Load models
emotion_model = load_model("emotion.h5", compile=False)
gender_age_model = load_model("gender_age_model.h5", compile=False)

emotion_labels = ['Angry','Happy','Sad','Surprise','Neutral']
gender_labels = ['Male','Female']


@app.get("/")
def home():
    return {"message": "Emotion, Gender and Age Detection API"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Emotion model input
    face_emotion = cv2.resize(gray,(48,48))
    face_emotion = face_emotion/255.0
    face_emotion = face_emotion.reshape(1,48,48,1)

    emotion_pred = emotion_model.predict(face_emotion)
    emotion = emotion_labels[np.argmax(emotion_pred)]

    # Gender Age model input
    face_gender = cv2.resize(img,(128,128))
    face_gender = face_gender/255.0
    face_gender = face_gender.reshape(1,128,128,3)

    pred = gender_age_model.predict(face_gender)

    gender = gender_labels[np.argmax(pred[0])]
    age = int(pred[1][0])

    return {
        "emotion": emotion,
        "gender": gender,
        "age": age
    }