import numpy as np
import cv2
import facerecognition as fr

faces,faceID=fr.label_training(r'path to your images')
face_recognizer=fr.trainClassifier(faces,faceID)
face_recognizer.save('modelname.yml')
name={0:'your name'}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+w,x:x+h]
    label,confidance=face_recognizer.predict(roi_gray)
    fr.draw_Rectangle(test_img,face)
    predict_name=name[label]
    fr.put_text(test_img,predict_name,x,y)
resized_image=cv2.resize(test_img,(1000,7000))
cv2.imshow('face',resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
