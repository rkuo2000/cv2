# openCV read test.jpg and resize the image to 2X
import cv2

img = cv2.imread('messi5.jpg')
height,width = img.shape[:2]

res = cv2.resize(img, (2*width, 2*height), interpolation = cv2.INTER_CUBIC)

cv2.imshow('Image' ,img)
cv2.imshow('Resize', res)


cv2.waitKey(0)
cv2.destroyAllWindows()
