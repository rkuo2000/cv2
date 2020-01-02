import numpy as np
import cv2

bg = cv2.imread('texture.jpg')
(bg_h, bg_w, bg_ch) = bg.shape
print(bg_h, bg_w, bg_ch)

img1 = cv2.imread('./croissant/croissant-0.jpg')
(img1_h, img1_w, img1_ch) = img1.shape
print(img1_h, img1_w, img1_ch)
cv2.imshow('img1', img1)
    
# Range of Color
lower_white = np.array([240,240,240])
upper_white = np.array([255,255,255])
	
mask = cv2.inRange(img1, lower_white, upper_white)
cv2.imshow('mask1', mask)

roi = bg[0:img1_h, 0:img1_w]
roi = cv2.bitwise_and(roi, img1, mask=mask)
cv2.imshow('roi1',roi)

roi = roi+img1
cv2.imshow('roi2',roi)

bg[0:img1_h, 0:img1_w]=roi
cv2.imshow('image', bg)

cv2.waitKey(0)
cv2.destroyAllWindows()
