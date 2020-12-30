import cv2
import sys

if len(sys.argv)>1:
    filename = sys.argv[1]
else:
    filename = 'test.jpg'
	
img = cv2.imread(filename)

cv2.imshow('Image',img)
cv2.waitKey(0)
