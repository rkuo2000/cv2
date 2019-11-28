import cv2

image = cv2.imread('gesture.jpg')

gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite('gesture_gray.jpg', gray)
