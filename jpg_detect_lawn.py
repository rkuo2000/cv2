import numpy as np
import cv2
import random

img = cv2.imread('lawn1.jpg')
height, width, channels = img.shape
roi = img[int(height*0.3):height, 0:width]
cv2.imshow('ROI', roi)

hsv  = cv2.cvtColor(roi,cv2.COLOR_RGB2HSV)
lower_green = np.array([35,90,120])
upper_green = np.array([75,180,200])

mask = cv2.inRange(hsv, lower_green, upper_green)
res  = cv2.bitwise_and(roi, roi, mask=mask)
kernel = np.ones((5,5),np.uint8)
morph = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
cv2.imshow('MASK', mask)
cv2.imshow('RESULT', res)
cv2.imshow('MORPH', morph)

# print h,w,ch
h, w, ch = morph.shape
print(h, w, ch)
# print one pixel (closer to botton)
pixel = morph[h-10,w-10]
print(pixel)

# print status of a 100x100 block 
for i in range(0,h,100):
    for j in range(0,w,100):
        pixel=morph[i,j]
        if pixel[0]==0 and pixel[1]==0 and pixel[2]==0:
            print(i,j,'CLEAR')
        else:
            print(i,j,'GRASS')

cv2.waitKey(0)	
cv2.destroyAllWindows()
