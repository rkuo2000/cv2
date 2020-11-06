import cv2

cap = cv2.VideoCapture(0)
while 1:
    ret, frame = cap.read()
    print(frame.shape)

    cv2.imshow('CAM', frame)
    key=cv2.waitKey(100)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()
