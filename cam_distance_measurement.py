from imutils import paths
import numpy as np
import imutils
import cv2

cap = cv2.VideoCapture(0)

while (True):
	ret, frame = cap.read()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	_,cnts,_ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # draw Enclosing Circle
    for c in cnts:
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(frame, (int(x), int(y)), int(radius), (0,255,0), 1)
		
	cv2.imshow("frame", frame)
	cv2.waitKey(0)

cv2.destroyAllWindows()
