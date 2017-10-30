import cv2
import numpy as np

img = cv2.imread('c:/test.jpg')
rows,cols = img.shape[:2]

M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('Image' ,img)
cv2.imshow('Rotation', dst)


cv2.waitKey(0)
cv2.destroyAllWindows()
