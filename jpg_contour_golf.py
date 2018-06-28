import cv2
import numpy as np

image = cv2.imread('golf.jpg')

lower = np.array([127, 127, 127])
upper = np.array([255, 255, 255])

filtered = cv2.inRange(image, lower, upper)
cv2.imshow('filtered', filtered)
blurred  = cv2.GaussianBlur(filtered, (5, 5), 0)
cv2.imshow('blurred',blurred)

# find contours in the image
(_, cnts, _) = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(cnts) >0:
    cnt = sorted(cnts, key=cv2.contourArea, reverse = True)[0]
	
    #draw rotated bounding box around the contour
    #rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
    #cv2.drawContours(image, [rect], -1, (0,255,0), 2)
	
    #draw circle around the contour
    #((x,y), radius) = cv2.minEnclosingCircle(cnt)
    #cv2.circle(image, (int(x), int(y)), int(radius), (0,255,0), 1)
	
    #draw bounding rectangle around the contour
    (x,y,w,h) = cv2.boundingRect(cnt)
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 1)
	
cv2.imshow("image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
