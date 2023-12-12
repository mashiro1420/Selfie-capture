import cv2
from datetime import datetime
import os
cap = cv2.VideoCapture("smile3.mp4")
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                    "haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                     "haarcascade_smile.xml")

count=1
now = datetime.now()
dtime = now.strftime("%d-%m-%y_%H-%M")
path = './date'+str(dtime)
if not os.path.exists(path):
  os.mkdir(path)

while True:
    ret,img = cap.read()
    _,imgcon = cap.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayImg,1.3,3)
    
    for x,y,w,h in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
        smiles = smileCascade.detectMultiScale(grayImg,1.9,20)
        for sx,sy,sw,sh in smiles:
            cv2.rectangle(img,(sx,sy),(sx+sw,sy+sh),(0,0,255),5)
            print("Ảnh đã được lưu "+str(count))
            path=r'D:\Schoolwork\Nam 4 ky 1\T3 - Computer Vision\smile\date'+str(dtime)+'\img'+str(count)+'.jpg'
            cv2.imwrite(path,imgcon)
            count +=1
    simg = cv2.resize(img, (960, 540)) 
    cv2.imshow('Smile selfie',simg)
    
    if(cv2.waitKey(1) & 0xFF==ord('q')):
        break

cap.release()                                  
cv2.destroyAllWindows() 
