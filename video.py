# Usage: python3 video.py filename.mp4
import sys
import cv2
import numpy as np

if len(sys.argv)>1:
    if sys.argv[1].isdigit():
        cap = cv2.VideoCapture(int(sys.argv[1])) # argv[1] = 0 or 1
    else:
        cap = cv2.VideoCapture(sys.argv[1]) # argv[1] = filename
else:
        cap = cv2.VideoCapture('video/MOT16-03-raw.webm')

while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow("input", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
