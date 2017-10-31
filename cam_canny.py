import cv2

cap = cv2.VideoCapture(0)
while 1:
  ret, frame = cap.read()
  gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray,100,200)

  cv2.imshow('Edges', edges)
  cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
                   
