import cv2
import sys
import numpy as np

if len(sys.argv)>1:
    filename = sys.argv[1]
else:
    filename = "H03.png"

greenBGR = np.uint8([[[0,255,0]]])
print(cv2.cvtColor(greenBGR, cv2.COLOR_BGR2HSV))

image = cv2.imread(filename)
cv2.imshow('IMAGE', image)

frame = image.copy()
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# lower & upper bound for any color
lower_bound = np.array([  0,  90,  90])
upper_bound = np.array([255, 255, 255])

mask = cv2.inRange(hsv, lower_bound, upper_bound)
cv2.imshow('MASK', mask)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(frame, frame, mask=mask)
cv2.imshow('RESULT', res)

# Find Contours
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw All Contours
frame = image.copy()
cv2.drawContours(frame, contours, -1, (0,255,0), 3)
cv2.imshow('CONTOURS', frame)

# Finding The Largest Contour
contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
#frame = image.copy()
#cv2.drawContours(frame, biggest_contour, -1, (0,255,0), 3)
#cv2.imshow('OBJECTS', frame)

# Bounding Rectangle
x,y,w,h = cv2.boundingRect(biggest_contour)
cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
cX = int(x + w/2)
cY = int(y + h/2)
cv2.circle(image,(cX, cY), 5, (1,227,254), -1)
cv2.imshow('B-BOX', image)

print(x,y,w,h)
print(cX,cY)

cv2.waitKey(1)
cv2.destroyAllWindows()
