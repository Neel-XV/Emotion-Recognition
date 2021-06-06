from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear",
                3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = keras.models.load_model('model_3.h5')
font = cv2.FONT_HERSHEY_SIMPLEX


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        # for (x, y, w, h) in faces:
        #     fc = gray_fr[y:y+h, x:x+w]

        #     roi = cv2.resize(fc, (48, 48))
        #     pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

        #     cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
        #     cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        for (x, y, w, h) in faces:
            cv2.rectangle(fr, (x, y), (x + w, y + h), (0, 255, 0), 1)
            roi_gray = gray_fr[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(
                cv2.resize(roi_gray, (48, 48)), -1), 0)
            cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1,
                          norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            prediction = model.predict(cropped_img)
            cv2.putText(fr, emotion_dict[int(np.argmax(
                prediction))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            label = np.argmax(prediction)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
