import cv2

img = cv2.imread('c:/test.jpg')
height,width = img.shape[:2]

res = cv2.resize(img, (2*width, 2*height), interpolation = cv2.INTER_CUBIC)

cv2.imshow('Image' ,img)
cv2.imshow('Resize', res)


cv2.waitKey(0)
cv2.destroyAllWindows()
