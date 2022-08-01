import cv2
import numpy as np

filename = "rock_climbing.png"
image = cv2.imread(filename)

# BGR convert into YCbCr
imgYCrCb  = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)

## min & max YCbCr of Skin
min_YCrCb = np.array([  0,135, 85],np.uint8)
max_YCrCb = np.array([255,180,135],np.uint8)

# InRange
mask = cv2.inRange(imgYCrCb,min_YCrCb,max_YCrCb)

# Bitwise and
skin = cv2.bitwise_and(image, image, mask = mask)

# Display 
cv2.imshow('IMAGE', image)
cv2.imshow('SKIN', skin)

cv2.waitKey(0)
cv2.destroyAllWindows()
