import numpy as np
import cv2

cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
  
    height, width, channels = img.shape
    roi = img
    #roi = img[int(height*0.7):height, int(width*0.1):int(width*0.9)]
    #cv2.imshow('ROI', roi)

    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edge = cv2.Canny(blur, 10, 160)
    #cv2.imshow('BLUR', blur)
    cv2.imshow('EDGE', edge)

    _, cnts, hierarchy = cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # draw Bounding Rectangle
    for c in cnts:
        (x,y,w,h) = cv2.boundingRect(c)
        if (w*h>10000 and w*h<40000):
            print(x,y,w,h)
            cv2.rectangle(roi, (x,y), (x+w, y+h), (0,255,0), 1)

    cv2.imshow('contours', roi)

    cv2.waitKey(1)
	
cap.release()
cv2.destroyAllWindows()
