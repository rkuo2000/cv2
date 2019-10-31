import cv2

img = cv2.imread('opencv_logo.jpg')

blur = cv2.blur(img,(5,5))
median = cv2.medianBlur(img,5)
gaussian = cv2.GaussianBlur(img,(5,5),0)
bilateral = cv2.bilateralFilter(img,9,75,75)

cv2.imshow('blur', blur)
cv2.imshow('median', median)
cv2.imshow('gaussian', gaussian)
cv2.imshow('bilateral', bilateral)

cv2.waitKey(0)
cv2.destroyAllWindows()
