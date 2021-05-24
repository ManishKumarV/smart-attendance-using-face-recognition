import numpy as np
import cv2
import os

def faceDetection(input_img):
    gray_img=cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
    face_haar=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')
    faces=face_haar.detectMultiScale(gray_img,scaleFactor=1.3,minNeighbors=3)
    return faces,gray_img

def label_training(directory):
    faces=[]
    faceID=[]

    for path,subdirectory,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                continue
            id=os.path.basename(path)
            img_path=os.path.join(path,filename)
            print("img",img_path)
            print("id",id)
            test_img=cv2.imread(img_path)
            if test_img is None:
                print('Image not loaded')
                continue
            faces_rect,gray_img=faceDetection(test_img)
            if len(faces_rect)!=1:
                continue
            (x,y,w,h)=faces_rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces,faceID

def trainClassifier(faces,faceid):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(faceid))
    return face_recognizer

def draw_Rectangle(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(155,234,0),5)

def put_text(test_img,text,x,y):                                    
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,3,(255,0,0),6)