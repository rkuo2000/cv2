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
#    top, bottom, left, right = 20, 320, 208, 508 # 300x300
    roi = frame[top:bottom, left:right]          # region of interest
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)	
    frame[top:bottom, left:right] = roi_rgb

    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)	
    cv2.imshow("Cam", frame)
	
    keypress = cv2.waitKey(1) & 0xFF # keypress by user 
    if keypress == ord("q"): # press q to quit
        break	
    if keypress == ord(" "): # press space to save the captured image
        filename="up_"+str(count)+".jpg"
        print(filename)
        gesture = cv2.resize(roi_gray, (96, 96), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(filename, gesture)
        count+=1                      

# free up memory
camera.release()
cv2.destroyAllWindows()
