import numpy as np
import random
import cv2
import sys

minSize= int(sys.argv[2])
segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=0.5, k=300, min_size=minSize)

img = cv2.imread(sys.argv[1])
height, width, channels = img.shape

segment = segmentator.processImage(img)
seg_image = np.zeros(img.shape, np.uint8)

for i in range(np.max(segment)):
    y, x = np.where(segment == i)
    color = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
    for xi, yi in zip(x,y):
        seg_image[yi,xi] = color

result = cv2.addWeighted(img, 0.3, seg_image, 0.7, 0)

cv2.imshow('result', result)
cv2.imwrite(sys.argv[1].replace('.cut.jpg', '')+'.out.jpg', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
