import os
import cv2

cap = cv2.VideoCapture(0)
while 1:
  ret, frame = cap.read()

  cv2.imshow('image', frame)
  cv2.imwrite('image.jpg',frame)
  os.system('alpr image.jpg')
  cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
                   
