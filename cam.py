import sys
import cv2

if len(sys.argv)>1:
   filename = sys.argv[1]
else:
   filename = 0

if filename ==0:
    cap = cv2.VideoCapture(0) # webcam
else:
    cap = cv2.VideoCapture(filename) # video file
	
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); #default 640
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720); #default 480

while(cap.isOpened()):
    ret, frame = cap.read()
    #frame = cv2.flip(frame, 0) # vertical flip
    frame = cv2.flip(frame, 1) # horizontal flip
    print(frame.shape) 	#no need to flip for opencv-python 4.2.0.32)

    cv2.imshow('CAM', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
