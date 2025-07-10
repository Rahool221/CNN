from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'/Users/rahulyadav/Desktop/Emotion_detection_CNN/haarcascade_frontalface_default.xml')
classifier =load_model(r'/Users/rahulyadav/Desktop/Emotion_detection_CNN/model.h5')

emotion_labels = ['angry','disgust','fear','happy','neutral', 'sad', 'surprise']

cap = cv2.VideoCapture(0)



while True:

   

    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            # roi = roi_gray.astype('float')/255.0
            # roi = img_to_array(roi)
            # roi = np.expand_dims(roi,axis=0)

            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=-1)  # adds channel dimension at the end -> (48,48,1)
            roi = np.expand_dims(roi, axis=0)   # adds batch dimension -> (1,48,48,1)


            prediction = classifier.predict(roi)[0]
           
            print("Prediction vector:", prediction)
            print("Sum of predictions:", np.sum(prediction))
            print("Argmax index:", np.argmax(prediction))

            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()







