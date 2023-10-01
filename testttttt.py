import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from datetime import date
import random

path='/Users/priakanthp/facerecognition/Imagesattendance'
images=[]
cn = []
l= os.listdir(path)

del l[0]
print(l)

for cl in l:
    ci=cv2.imread(f'{path}/{cl}')
    images.append(ci)
    cn.append(os.path.splitext(cl)[0])
print(cn)

def findencodings(images):
    el =[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode= face_recognition.face_encodings(img)[0]
        el.append(encode)
    return el

def markattendance(name):
    with open('sa.csv','r+') as f:
        dl = f.readlines()
        print(dl)
        nl=[]
        for line in dl:
            entry=line.split(',')
            nl.append(entry[0])
        if name  not in nl:
            intime=datetime.now()
            outtime=datetime.now()
            now = datetime.now()
            today = date.today()
            d1 = today.strftime("%d/%m/%Y")
            dtString = now.strftime('%H:%M:%S')
            dtString1 = intime.strftime('%H:%M:%S')
            dtString2 = outtime.strftime('%H:%M:%S')
            x=random.randint(100,1000)
            f.writelines(f'\n{name},{d1},{dtString1},{x},{dtString2}')
#markattendance('risha')
lk= findencodings(images)
cap = cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=  cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    ff= face_recognition.face_locations(imgS)
    ef=face_recognition.face_encodings(imgS,ff)

    for encodeFace, faceLoc in zip(ef, ff):
        matches = face_recognition.compare_faces(lk,encodeFace)
        fd= face_recognition.face_distance(lk,encodeFace)
        print(fd)
        mi= np.argmin(fd)
        if matches[mi]:
            name= cn[mi].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1= y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markattendance(name)
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)


