import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import defaultdict
import pandas as pd
from PIL import Image

st.set_page_config(page_title="AI Face Analytics Dashboard")

st.title("Emotion + Age + Gender Detection")

# Load models
emotion_model = load_model("emotion.h5")
age_gender_model = load_model("age_gender_model.h5")

emotion_labels = ['Angry','Happy','Neutral','Sad','Surprise']
gender_labels = ['Male','Female']

emotion_counts = defaultdict(int)
gender_counts = defaultdict(int)

mode = st.sidebar.selectbox("Select Mode",["Webcam","Upload Image"])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
)

# -----------------------------
# WEBCAM MODE
# -----------------------------

if mode == "Webcam":

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])
    chart_area = st.empty()

    emotion="Detecting"
    gender="Detecting"
    age=0

    if run:

        cap = cv2.VideoCapture(0)

        cap.set(3,640)
        cap.set(4,480)

        frame_count = 0

        while True:

            ret,frame = cap.read()
            if not ret:
                break

            frame_count+=1

            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=face_cascade.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:

                face=frame[y:y+h,x:x+w]

                # emotion input
                roi=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                roi=cv2.resize(roi,(48,48))
                roi=roi.reshape(1,48,48,1)/255.0

                # gender/age input
                ag_face=cv2.resize(face,(64,64))
                ag_face=ag_face.reshape(1,64,64,3)/255.0

                if frame_count%10==0:

                    # Emotion
                    emotion_pred=emotion_model.predict(roi,verbose=0)
                    emotion=emotion_labels[np.argmax(emotion_pred)]
                    emotion_counts[emotion]+=1

                    # Age + Gender
                    ag_pred=age_gender_model.predict(ag_face,verbose=0)

                    gender=gender_labels[np.argmax(ag_pred[1])]
                    age=int(ag_pred[0][0])

                    gender_counts[gender]+=1

                label=f"{emotion} | {gender} | Age:{age}"

                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,label,(x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,(0,255,0),2)

            FRAME_WINDOW.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

            if sum(emotion_counts.values())>0:

                col1,col2=chart_area.columns(2)

                with col1:
                    st.subheader("Emotion Chart")
                    df1=pd.DataFrame(emotion_counts.items(),
                                     columns=["Emotion","Count"])
                    st.bar_chart(df1.set_index("Emotion"))

                with col2:
                    st.subheader("Male vs Female")
                    df2=pd.DataFrame(gender_counts.items(),
                                     columns=["Gender","Count"])
                    st.bar_chart(df2.set_index("Gender"))

        cap.release()

# -----------------------------
# IMAGE UPLOAD MODE
# -----------------------------

else:

    uploaded_file=st.file_uploader(
        "Upload Image",
        type=["jpg","png","jpeg"]
    )

    if uploaded_file:

        image=Image.open(uploaded_file)
        frame=np.array(image)

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:

            face=frame[y:y+h,x:x+w]

            roi=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            roi=cv2.resize(roi,(48,48))
            roi=roi.reshape(1,48,48,1)/255.0

            ag_face=cv2.resize(face,(64,64))
            ag_face=ag_face.reshape(1,64,64,3)/255.0

            emotion_pred=emotion_model.predict(roi,verbose=0)
            emotion=emotion_labels[np.argmax(emotion_pred)]

            ag_pred=age_gender_model.predict(ag_face,verbose=0)

            gender=gender_labels[np.argmax(ag_pred[1])]
            age=int(ag_pred[0][0])

            label=f"{emotion} | {gender} | Age:{age}"

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,label,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,(0,255,0),2)

        st.image(frame)