import sys
import numpy as np
import cv2

filename = sys.argv[1]
image = cv2.imread(filename)

rows,cols = image.shape[:2]
print(rows, cols)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (11,11), 0)
edge = cv2.Canny(blur, 20, 160)
cv2.imshow("edges", edge)

(cnts, hierarchy) = cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# detect max_area of contours
max_ar = 0
for c in cnts:
    ar = cv2.contourArea(c)
    if ar>max_ar: 
        max_ar = ar
        max_c  = c

box = cv2.minAreaRect(max_c)
box = np.int0(cv2.boxPoints(box))
print(box)

cv2.drawContours(image, [box], -1, (0,255,0), 2)
cv2.imshow("contours", image)
cv2.imwrite("output.jpg", image)

cv2.waitKey(0)
cv2.destroyAllWindows()

