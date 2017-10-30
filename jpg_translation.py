import cv2
import numpy as np

img = cv2.imread('c:/test.jpg')
rows,cols = img.shape[:2]
M = np.float32([[1,0,100],[0,1,50]])
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('Image' ,img)
cv2.imshow('Translation', dst)


cv2.waitKey(0)
cv2.destroyAllWindows()
