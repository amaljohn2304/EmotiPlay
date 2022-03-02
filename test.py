'''
PyPower Projects
Emotion Detection Using AI
'''

#USAGE : python test.py

from playsound import playsound
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import winsound
import random

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier =load_model('./Emotion_Detection.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

cap = cv2.VideoCapture(0)

hp=0
sd=0
an=0
ne=0
label='hh'
while True:
    n=random.randint(1,4)
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0]
            print("\nprediction = ",preds)
            label=class_labels[preds.argmax()]
            print("\nprediction max = ",preds.argmax())
            print("\nlabel = ",label)
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        print("\n\n")
    if cv2.waitKey(1) & 0xFF == ord('n'):
        hp=0
        sd=0
        an=0
        ne=0
    if(label=='Happy' and hp==0):
        winsound.PlaySound(f"music\happy\happy{n}.wav", winsound.SND_ASYNC | winsound.SND_ALIAS )
        hp=1
        sd=0
        an=0
        ne=0
    if(label=='Neutral' and ne==0):
        winsound.PlaySound(f"music\\neutral\\neut{n}.wav", winsound.SND_ASYNC | winsound.SND_ALIAS )
        ne=1
        hp=0
        an=0
        sd=0
    if(label=='Sad' and sd==0):
        winsound.PlaySound(f"music\sad\sad{n}.wav", winsound.SND_ASYNC | winsound.SND_ALIAS )
        sd=1
        hp=0
        an=0
        ne=0
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


























