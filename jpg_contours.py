import numpy as np
import cv2

img = cv2.imread('test.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (11,11), 0)
edge = cv2.Canny(blur, 20, 160)
(cnts, hierarchy) = cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# Draw all contours
cv2.drawContours(img, cnts, -1, (0,255,0), 1)

# draw center in yellow-dot
for c in cnts:
    M= cv2.moments(c)
    cX = int(M["m10"]/M["m00"])
    cY = int(M["m01"]/M["m00"])
    cv2.circle(img,(cX,cY), 5, (1,227,254), -1)

cv2.imshow("contours", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
