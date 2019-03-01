import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

# set orange thresh
lower_orange=np.array([0,70,100])
upper_orange=np.array([80,255,255])

while(1):
    # get a frame and show
    ret, frame = cap.read()
    #cv2.imshow('Capture', frame)

    # change to hsv model
    hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv2,(11,11),0)
	
    # get mask
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    edge = cv2.Canny(mask, 20, 160)
    cnts, hierarchy = cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
      M= cv2.moments(c)
      if M["m00"] != 0:
        cX = int(M["m10"]/M["m00"])
        cY = int(M["m01"]/M["m00"])
        area = cv2.contourArea(c)
        if area > 800:
          cv2.circle(frame,(cX,cY), 5, (1,1,254), -1)
          print(cX,cY)
    #cv2.imshow('Mask', edge)
    # detect orange
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('Result', res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
                   
