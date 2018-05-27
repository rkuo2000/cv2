import numpy as np
import cv2
import sys

img  = cv2.imread(sys.argv[1])

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

blur = cv2.GaussianBlur(gray, (3,3), 0)

retval,thresh = cv2.threshold(blur, 125, 255, 0)

img2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img2, contours, -1, (0,255,0), 3)

cv2.imshow('org', img)
cv2.imshow('contours', img2)

cv2.waitKey(10000)
cv2.destroyAllWindows()
