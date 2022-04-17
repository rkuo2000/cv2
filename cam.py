import cv2
import sys
if len(sys.argv) >1:
    vid = int(sys.argv[1])
else:
    vid = 0
cap = cv2.VideoCapture(vid)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720);

while(cap.isOpened()):
    ret, frame = cap.read()
    print(frame.shape)
    #frame = cv2.flip(frame, 1) # 0: vertical flip, 1: horizontal flip

    cv2.imshow('Camera', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
