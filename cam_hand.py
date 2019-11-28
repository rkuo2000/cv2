
import cv2
import numpy as np

camera = cv2.VideoCapture(0)

count = 0

while(True):
    # get the current frame
    (_, frame) = camera.read()
    (height, width, channel) = frame.shape       # frame shape
	 
    top, right, bottom, left = 10, 350, 234, 574 # 224x224
    roi = frame[top:bottom, right:left]          # region of interest
        
    frame = cv2.flip(frame, 1) # flip the frame so that it is not the mirror view
	
    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
    cv2.imshow("Cam", frame)
    print("Frame_No: ",count)
	
    if (count%120 ==0):
        cv2.imwrite("hand_"+str(count)+".jpg", roi)
    count+=1
	
    keypress = cv2.waitKey(1) & 0xFF # keypress by user    
    if keypress == ord("q"):         # if the user pressed "q",
        break                        # then stop looping

# free up memory
camera.release()
cv2.destroyAllWindows()
