import cv2
import sys
import numpy as np

if len(sys.argv) >1:
    vid = int(sys.argv[1])
else:
    vid = 0
cap = cv2.VideoCapture(vid)

while(cap.isOpened()):
    ret, frame = cap.read()
    #frame = cv2.flip(frame, 1) # 0: vertical flip, 1: horizontal flip

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # lower & upper bound for any color
    lower_bound = np.array([  0,  90,  90])
    upper_bound = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)
    #cv2.imshow('RESULT', res)

    # Find Contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Finding The Largest Contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    # Bounding Rectangle
    x,y,w,h = cv2.boundingRect(biggest_contour)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cX = int(x + w/2)
    cY = int(y + h/2)
    cv2.circle(frame,(cX, cY), 5, (1,227,254), -1)
    cv2.imshow('Bounding BOX', frame)

    print(x,y,w,h)
    print(cX,cY)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

