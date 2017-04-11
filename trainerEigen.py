import posixpath
import os
import cv2
import numpy as np
from PIL import Image

#recognizer=cv2.face.createEigenFaceRecognizer();
path="/home/shivani/Desktop/faces"
def getImagesWithID(path):
    imagePaths=[posixpath.join(path,f) for f in os.listdir(path)]
    facesh=[]
    IDs=[]
    for imagePath in imagePaths:
        #print(imagePath)
        faceImg=Image.open(imagePath).convert('L')
        faceNp=np.array(faceImg,'uint8')
        #faceNp=np.array(miniframe,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        #ID=str(os.path.split(imagePath)[-1].split('.')[1])
        facesh.append(faceNp)
        #print(ID)
        IDs.append(ID)
        #cv2.imshow("training",faceNp)
        #cv2.waitKey(10)
    return IDs,facesh


def facerec():
    names=['Shivi','Unknown']
    Ids,facesh=getImagesWithID(path)
    recognizer = cv2.face.createEigenFaceRecognizer()
    recognizer.train(facesh,np.array(Ids))
    #recognizer.save('/home/shivani/opencv-3.2.0/recognizer/trainingData.yml')
    #recognizer.load('trainer/trainingData.yml')
    cascadePath = "/home/aparna/Downloads/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);
    cam = cv2.VideoCapture(0)
    #font = cv2.InitFont(cv2.FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
    while True:
        ret,im =cam.read()
        #print(im.shape[0],im.shape[1])
        minisize = (im.shape[1],im.shape[0])
        miniframe = cv2.resize(im, minisize)
        #print(im.shape[0],im.shape[1])
        gray=cv2.cvtColor(miniframe,cv2.COLOR_BGR2GRAY)
        #gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.3,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            #f=cv2.resize(gray[y:y+h,x:x+w],(200,200))
            f=cv2.resize(gray[y:y+h,x:x+w],(200,200))
            Id, conf = recognizer.predict(f)
            #print(h)
            cv2.putText(im,str(Id), (x+5,y+h-20),cv2.FONT_HERSHEY_SIMPLEX,2, 255,3)
        cv2.imshow('im',im) 
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

#a=trainerEigen()
facerec()
