import numpy as np
import cv2
import sys
if len(sys.argv)>1:
    img = cv2.imread(sys.argv[1])
else:
    img = cv2.imread("Hiwin-RT605-71-GB.jpg")

height, width, channels = img.shape
cv2.imshow('img', img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
edge = cv2.Canny(blur, 20, 160)
cv2.imshow('BLUR', blur)
cv2.imshow('EDGE', edge)

cnts, hierarchy = cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# draw Bounding Rectangle
for c in cnts:
    (x,y,w,h) = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    print(area, x, y, w, h)
    if area > int(sys.argv[2]):
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)

cv2.imshow('contours', img)

# draw Area Rectangle
#for c in cnts:
#    box = cv2.minAreaRect(c)
#    box = np.int0(cv2.boxPoints(box))
#    cv2.drawContours(img, [box], -1, (0,255,0), 1)


cv2.waitKey(0)
cv2.destroyAllWindows()
