import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    ret, frame = cap.read()

    #imgHSV   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #min_HSV   = np.array([ 0,  15,   0], dtype = "uint8")
    #max_HSV   = np.array([17, 105, 255], dtype = "uint8")

    imgYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    min_YCrCb = np.array([  0,135, 85], np.uint8)
    max_YCrCb = np.array([255,180,135], np.uint8)

    # InRange
    #mask = cv2.inRange(imgHSV, min_HSV, max_HSV)
    mask = cv2.inRange(imgYCrCb, min_YCrCb, max_YCrCb)

    # Bitwise and
    skin = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("images", np.hstack([frame, skin]))

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
