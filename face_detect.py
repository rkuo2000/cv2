# Usage: python face_detect.py friends.jpg
import numpy as np
import cv2
import sys

if len(sys.argv)>1:
    img = cv2.imread(sys.argv[1])
else:
    img = cv2.imread("friends.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

bboxes = face.detectMultiScale(gray)

for box in bboxes:
    print(box)
    (x,y,w,h) = tuple(box)
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
