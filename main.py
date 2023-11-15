import cv2

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier("dataset/haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier("dataset/haarcascade_smile.xml")

while True:
    ret,img = cap.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayImg,1.3,3)
    count=1

    for x,y,w,h in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
        smiles = smileCascade.detectMultiScale(grayImg,1.8,15)
        for sx,sy,sw,sh in smiles:
            cv2.rectangle(img,(sx,sy),(sx+sw,sy+sh),(0,0,255),5)
            print("Ảnh "+str(count)+" đã được lưu")
            path=r'D:\Schoolwork\Nam 4 ky 1\T3 - Computer Vision\smile\img'+str(count)+'.jpg'
            cv2.imwrite(path,img)
            count +=1
            if(count>3):   
                break
                
    cv2.imshow('Smile selfie',img)
    if(cv2.waitKey(1) & 0xFF==ord('q')):
        break

cap.release()                                  
cv2.destroyAllWindows() 
