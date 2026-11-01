import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load models
emotion_model = load_model("emotion.h5", compile=False)
gender_age_model = load_model("gender_age_model.h5", compile=False)

emotion_labels = ['Angry','Happy','Sad','Surprise','Neutral']
gender_labels = ['Male','Female']

# Face detector
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        face_img = frame[y:y+h,x:x+w]
        face_gray = gray[y:y+h,x:x+w]

        # Emotion prediction
        em = cv2.resize(face_gray,(48,48))/255.0
        em = em.reshape(1,48,48,1)

        emotion = emotion_labels[
            np.argmax(emotion_model.predict(em))
        ]

        # Gender + Age prediction
        ga = cv2.resize(face_img,(128,128))/255.0
        ga = ga.reshape(1,128,128,3)

        pred = gender_age_model.predict(ga)

        gender = gender_labels[np.argmax(pred[0])]
        age = int(pred[1][0])

        label = f"{emotion} | {gender} | {age}"

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,label,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    cv2.imshow("Real-Time Detection",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()