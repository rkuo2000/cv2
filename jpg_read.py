import cv2
import sys

if len(sys.argv)>1:
    img = cv2.imread(sys.argv[1])
else:
    img = cv2.imread('test.jpg')
cv2.imshow('Image',img)
cv2.waitKey(0)
