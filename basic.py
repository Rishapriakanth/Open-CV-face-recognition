import cv2
import numpy as np
import face_recognition
imgElon= face_recognition.load_image_file('/Users/priakanthp/facerecognition/Images/Elon Musk.jpeg')
imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgtest= face_recognition.load_image_file('/Users/priakanthp/facerecognition/Images/Bill gates.jpg')
imgtest=cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)


faceLoc= face_recognition.face_locations(imgElon)[0]
encodeElon=face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[2]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
print(faceLoc)

faceLoctest= face_recognition.face_locations(imgtest)[0]
encodetest=face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLoctest[3],faceLoctest[2]),(faceLoctest[1],faceLoctest[2]),(255,0,255),2)

results= face_recognition.compare_faces([encodeElon],encodetest)
faceDis= face_recognition.face_distance([encodeElon],encodetest)
print(results)
cv2.putText(imgtest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon test',imgtest)
cv2.waitKey(0)
