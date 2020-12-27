# openCV color space conversion into Gray/HSV, and converted from Gray/HSV to RGB
import cv2

img = cv2.imread('test.jpg')
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
org1= cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
org2= cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cv2.imshow('ORG' ,img)
cv2.imshow('GRAY',gray)
cv2.imshow('HSV' ,hsv)
cv2.imshow('ORG1',org1)
cv2.imshow('ORG2',org2)

cv2.waitKey(0)
cv2.destroyAllWindows()
