import numpy as np
import cv2

for i in range(10):
DATA_PATH= './bread/croissant'
FILENAME = 'images-0.jpg'
file = DATA_PATH+'/'+FILENAME

img = cv2.imread(file)
height, width, channels = img.shape

#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#hsv  = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

# Range of Color
lower_white = np.array([200,200,200])
upper_white = np.array([255,255,255])

mask = cv2.inRange(img, lower_white, upper_white)
mask = cv2.bitwise_not(mask)

x, y, w, h = cv2.boundingRect(mask)
#bbox = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
#cv2.imshow('image', img)

img = img[y:y+h, x:x+w]
mask = cv2.inRange(img, lower_white, upper_white)
mask = cv2.bitwise_not(mask)
cv2.imshow('mask', mask)
cv2.imwrite(DATA_PATH+'/croissant-0-mask.jpg', mask)

res = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow('result', res)
cv2.imwrite(DATA_PATH+'/croissant-0.jpg', res)

cv2.waitKey(0)
cv2.destroyAllWindows()

