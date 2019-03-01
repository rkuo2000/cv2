import numpy as np
import cv2

img = cv2.imread('test.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (11,11), 0)
edge = cv2.Canny(blur, 20, 160)
(cnts, hierarchy) = cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# draw Area Rectangle
for c in cnts:
    box = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(box))
    cv2.drawContours(img, [box], -1, (0,255,0), 1)

cv2.imshow("contours", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
