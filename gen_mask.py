# Usage: python gen_mask.py celeba_00.jpg
# input : examples/celeba/images/celeba_00.jpg
# output: examples/celeba/masks/celeba_00.jpg
import sys
import numpy as np
import cv2

# manually edit put white mask on image first
imagePath = 'examples/celeba/images/'

maskPath  = 'examples/celeba/masks/'
filename = sys.argv[1]

img = cv2.imread(imagePath+filename)

# Range of Color
lower_white = np.array([245,245,245])
upper_white = np.array([255,255,255])
	
mask = cv2.inRange(img, lower_white, upper_white)
	
cv2.imshow('image', img)
cv2.imshow('mask', mask)

cv2.imwrite(maskPath+filename,mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
