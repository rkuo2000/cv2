import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(1):
    # Take each frame
    _, frame = cap.read()

    height,width,channels = frame.shape
    roi = frame
    #roi = frame[int(height*0.7):height, int(width*0.1):int(width*0.9)]
    cv2.imshow('ROI', roi)
	
    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edge = cv2.Canny(blur, 20, 160)    
    cv2.imshow('EDGE', edge)
	
    cnts, hierarchy = cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
    for c in cnts:
        (x,y,w,h) = cv2.boundingRect(c)        
        area=w*h
        if(area>10000 ):
            continue
        cv2.rectangle(roi, (x,y), (x+w, y+h), (0,255,0), 3)
        cv2.circle(roi, (int(x+w/2),int(y+h/2)), 5, (0,0,255), 5)
        print("x",x+w/2,"y",y+h/2)      
        cv2.imshow('contours', roi)  
	
  
    k = cv2.waitKey(5) & 0xFF
    if k==27:
        break
	
cv2.destroyAllWindows()
