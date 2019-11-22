# Usage: python resize.py 1.jpg
# Output: resized.jpg
import cv2
import sys

filename = sys.argv[1]
img = cv2.imread(filename)
height,width = img.shape[:2]

print("Shrink image size to 50% !")
percentage = 0.5
res = cv2.resize(img, (int(width*percentage), int(height*percentage)), interpolation = cv2.INTER_CUBIC)
cv2.imshow('Resize', res)
cv2.imwrite('resized.jpg', res)

cv2.waitKey(0)
cv2.destroyAllWindows()
