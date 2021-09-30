# openCV read test.jpg and generate Sobel effects
import cv2

org = cv2.imread('test.jpg')

# convert to gray
gray  = cv2.cvtColor(org, cv2.COLOR_RGB2GRAY)

# remove noise
img   = cv2.GaussianBlur(gray, (3,3), 0)

# convolute with proper kernels
laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobel_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=5)
sobel_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=5)
cv2.imshow('Laplacian', laplacian)
cv2.imshow('SobelX', sobel_x)
cv2.imshow('SobelY', sobel_y)

abs_grad_x = cv2.convertScaleAbs(sobel_x)
abs_grad_y = cv2.convertScaleAbs(sobel_y)
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)    
cv2.imshow('Sobel', grad)

cv2.waitKey(0)
cv2.destroyAllWindows()
