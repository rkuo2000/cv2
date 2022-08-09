import numpy as np
import sys
import cv2

if len(sys.argv)>1:
    filename = sys.argv[1]
else:
    filename = "H03.png"

image = cv2.imread(filename)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (11,11), 0)
cv2.imshow("Blur", blur)

edge = cv2.Canny(blur, 30, 160)
cv2.imshow("Edge", edge)

(cnts, hierarchy) = cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# Draw all contours
cv2.drawContours(image, cnts, -1, (0,255,0), 1)

# draw center in yellow-dot
for c in cnts:
    M= cv2.moments(c)
    cX = int(M["m10"]/M["m00"])
    cY = int(M["m01"]/M["m00"])
    print(cX, cY)
    cv2.circle(image,(cX,cY), 5, (1,227,254), -1)

cv2.imshow("contours", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
