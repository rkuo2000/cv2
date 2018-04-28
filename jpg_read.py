# openCV read test.jpg and display the image file
import cv2

img = cv2.imread('test.jpg')
cv2.imshow('TEST',img)
cv2.waitKey(0)
