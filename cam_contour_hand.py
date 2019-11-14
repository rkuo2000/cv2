import cv2
import numpy as np
import imutils

if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5

    cap = cv2.VideoCapture(0)
    top, right, bottom, left = 10, 350, 225, 590 # ROI coordinates
    num_frame = 0

    while(True):
        (grabbed, frame) = cap.read()
        #frame = imutils.resize(frame, width=700)
		
        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        blur = cv2.GaussianBlur(gray, (5,5), 0)		
        #thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.threshold(blur, 230, 255, cv2.THRESH_BINARY_INV)[1]
        cv2.imshow("thresh", thresh)

        cnts = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(roi, [c], -1, (0, 255, 255), 2)
        frame[top:bottom, right:left]=roi
        cv2.imshow("frame", frame)
        print("frame no.", num_frame)
        num_frame += 1
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"): # press q to stop
            break

cap.release()
cv2.destroyAllWindows()
