import cv2

org = cv2.imread('test.png')

gray  = cv2.cvtColor(org, cv2.COLOR_RGB2GRAY)

# remove noise
img   = cv2.GaussianBlur(gray, (3,3), 0)

# convolute with proper kernels
laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobelx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=5)

cv2.imshow('Laplacian', laplacian)
cv2.imshow('SobelX', sobelx)
cv2.imshow('SobelY', sobely)

cv2.waitKey(0)
cv2.destroyAllWindows()
