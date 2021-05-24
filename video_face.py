import numpy as np
import cv2
import os
import facerecognition as fr
print (fr)
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('savedmodel.yml')
cap=cv2.VideoCapture(2)
name={0:"your name"}
while True:
    ret,test_img=cap.read()
    faces_detected,gray_img=fr.faceDetection(test_img)
    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+h,x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)
        print ("Confidence :",confidence)
        print("label :",label)
        fr.draw_Rectangle(test_img,face)
        predicted_name=name[label]
        fr.put_text(test_img,predicted_name,x,y)

    resized_img=cv2.resize(test_img,(1000,700))

    cv2.imshow("face detection ", resized_img)
    if cv2.waitKey(10)==ord('q'):
        break
