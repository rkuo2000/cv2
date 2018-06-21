import numpy as np
import cv2

im = cv2.imread('test.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# Draw all contours
#img = cv2.drawContours(img, contours, -1, (0,255,0), 3)

# Draw 4th contours
#img = cv2.drawContours(img, contours, 3, (0,255,0), 3)

# Draw one contours
#cnt = contours[4]
#img = cv2.drawContours(img, [cnt], 0, (0,255,0), 3)

print(len(contours))
#for i in range(len(contours)):
#    img = cv2.drawContours(image, contours, i, (0,255,0), 3)
#    cv2.imshow('Contour'+str(i), img)