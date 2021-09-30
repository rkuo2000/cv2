import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print(frame.shape)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    print(hsv[240,320])
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 240, 240], dtype = "uint8")
    #lower = np.array([ 0,  48, 100], dtype = "uint8")
    #upper = np.array([30, 100, 220], dtype = "uint8")	
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.erode(mask, kernel, iterations = 2)
    mask = cv2.dilate(mask, kernel, iterations = 2)

    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask = mask)

    cv2.circle(frame, (320,240), 5, (0,255,0), 1)
    cv2.imshow("images", np.hstack([frame, skin]))

    if cv2.waitKey(100) & 0xFF == ord(' '):
        cv2.imwrite("isolated_skin.jpg", np.hstack([frame, skin]))

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()