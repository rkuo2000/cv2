# Camera capture a gesture 
# roi = 224x224, convert to gray, and resize it to 96x96
import cv2

camera = cv2.VideoCapture(0)

(_, frame) = camera.read()
(height, width, channel) = frame.shape
print(width, height)

count = 0

while(True):
    # get the current frame
    (_, frame) = camera.read()
    frame = cv2.flip(frame, 1) # flip the frame so that it is not the mirror view
	
    top, bottom, left, right = 20, 244, 208, 432 # 224x224
    roi = frame[top:bottom, left:right]          # region of interest
        	
    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
	
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ROI  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    frame[top:bottom, left:right] = ROI
	
    cv2.imshow("Cam", frame)
	
    keypress = cv2.waitKey(1) & 0xFF # keypress by user 
    if keypress == ord("q"): # press q to quit
        break	
    if keypress == ord(" "): # press space to save the captured image
        filename="right_"+str(count)+".jpg"       
        print(filename)
        res = cv2.resize(gray, (96, 96), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(filename, res)
        count+=1                      

# free up memory
camera.release()
cv2.destroyAllWindows()
