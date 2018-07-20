import numpy as np
import cv2

#img = cv2.imread('lawn.jpg')
img = cv2.imread('field.jpg')
cv2.imshow('IMG', img)
height, width, channels = img.shape
roi = img[int(height*0.65):height, int(width*0.1):int(width*0.9)]
cv2.imshow('ROI', roi)

gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
edge = cv2.Canny(blur, 20, 160)
cv2.imshow('BLUR',blur)
cv2.imshow('EDGE',edge)
_, cnts, hierarchy = cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(roi, cnts, -1, (0,255,0), 1)
cv2.imshow("contours", roi)

cv2.waitKey(0)
cv2.destroyAllWindows()
