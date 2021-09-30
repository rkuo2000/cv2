import cv2

img = cv2.imread('test.jpg')

src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(src)

cv2.imshow('Source', src)
cv2.imshow('Equalized', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
