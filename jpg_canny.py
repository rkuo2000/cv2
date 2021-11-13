# openCV read test.jpg, color converted to gray, then run OpenCV canny to generate edge detection
import cv2
import sys

if len(sys.argv)>1:
    filename = sys.argv[1]
else:
    filename = 'messi5.jpg'

image = cv2.imread(filename)

gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray,100,200)
cv2.imshow('Gray', gray)
cv2.imshow('Edges', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
