import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image = cv2.imread('test.png')
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

for (x,y,w,h) in faces:
	cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

cv2.imshow('FACES' ,image)

cv2.waitKey(0)
cv2.destroyAllWindows()
