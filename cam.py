import cv2
#cap.open('xxxx.mp4') # for open mp4 file

cap = cv2.VideoCapture(0) # webcam #0
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); #default 640
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720); #default 480
print(cap.isOpened())

while 1:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 0) # vertical flip (上下顛倒)
    frame = cv2.flip(frame, 1) # horizontal flip (左右交換=鏡面效果)
    print(frame.shape)

    cv2.imshow('CAM', frame)
    key=cv2.waitKey(100)
    if key==27: # press [Esc] key
        break

cap.release()
cv2.destroyAllWindows()
