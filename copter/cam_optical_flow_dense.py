# Blog: [OpenCV Optical Flow介紹](https://meetonfriday.com/posts/e2795e5a/)
import sys
import cv2
import numpy as np

if len(sys.argv) >1:
    vid = int(sys.argv[1])
else:
    vid = 0
cap = cv2.VideoCapture(vid)

ret, first_frame = cap.read()
first_frame = cv2.flip(first_frame, 1) # horizontal flip for mirror effect
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(first_frame)
mask[..., 1] = 255

while(cap.isOpened()):
    ret, frame = cap.read()
    #cv2.imshow("input", frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    cv2.imshow("optical-flow dense", bgr)
    prev_gray = gray
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
