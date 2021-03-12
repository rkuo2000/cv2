import numpy as np
import random
import cv2
import sys

X1 = int(sys.argv[3])
Y1 = int(sys.argv[4])
X2 = int(sys.argv[5])
Y2 = int(sys.argv[6])

minSize= int(sys.argv[2])
segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=0.5, k=300, min_size=minSize)

img = cv2.imread(sys.argv[1])
height, width, channels = img.shape

roi = img[Y1:Y2, X1:X2]
segment = segmentator.processImage(roi)
seg_image = np.zeros(roi.shape, np.uint8)

for i in range(np.max(segment)):
    y, x = np.where(segment == i)
    color = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
    for xi, yi in zip(x,y):
        seg_image[yi,xi] = color

result = cv2.addWeighted(roi, 0.3, seg_image, 0.7, 0)
img[Y1:Y2, X1:X2] = result

cv2.imshow('img', img)
cv2.imwrite(sys.argv[1].replace('.jpg', '')+'.out.jpg', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
