import matplotlib.pyplot as plt
import numpy as np
import pyzbar.pyzbar as pyzbar
import cv2


#path definition
path = '/home/fajrin/CV/bola.jpg'
path1 = '/home/fajrin/CV/jj.JPG'
path2 = '/home/fajrin/CV/pres.jpeg'
path3 = '/home/fajrin/CV/tni.jpg'
path4 = '/home/fajrin/CV/ilmuwan.jpeg'
path5 = '/home/fajrin/CV/QR.png'
path6 = '/home/fajrin/CV/qrs.jpg'
cascade_path = '/home/fajrin/CV/facedetection-master/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

def show_image (img):
    plt.figure(figsize=(15,12))
    plt.imshow(cv2.cvtColor(img , cv2.COLOR_BGR2RGB))
    plt.title('IMAGE')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def image_detect(img):
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(img , 1.2 ,4)
    for (x,y,w,h) in faces :
        cv2.rectangle(img ,(x,y) , (x+w , y+h) , (0,0,255),3)
    show_image(img)

def frame_detect (cap):
    while True :
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray , 1.2 , 4)
        for (x,y,h,w) in faces :
            cv2.rectangle(frame , (x,y) ,(x+w, y+h),(0,255,0),3)
        cv2.imshow('frame' , frame)
        if cv2.waitKey(30)&0xFF == ord('q'):
            break
    cap.release()

def image_qr (img):
    decodeObjects = pyzbar.decode(img)
    for obj in decodeObjects:
        print(obj.data)
        cv2.putText(img,str(obj.data),(50,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
    show_image(img)

def frame_qr (cap):
    while True:
        _,frame = cap.read()
        decodeObject = pyzbar.decode(frame)
        for obj in decodeObject :
            cv2.putText(frame,str(obj.data), (50,50), cv2.FONT_HERSHEY_PLAIN,3,(255,255,0),2)
            print(obj.data)
        cv2.imshow('frame',frame)
        k = cv2.waitKey(1)
        if k == 27 :
            break

#image_qr(cv2.imread(path6))
#pic_qr(cv2.imread(path5))
cap= cv2.VideoCapture(0)
frame_qr(cap)
cv2.destroyAllWindows()