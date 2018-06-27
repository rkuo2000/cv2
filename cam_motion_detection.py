import cv2

#videoFile = "my_video.mp4"
#outputFolder = "my_output"
#if not os.path.exists(outputFolder):
#    os.makedirs(outputFolder)

# 開啟影片檔
#cap = cv2.VideoCapture(videoFile)
cap = cv2.VideoCapture(0)

#width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#height= cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#area = width * height

while(True):
    ret, frame = cap.read()
	
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray, (21,21), 0)
	
    if firstFrame is None:
        firstFrame = gray
        continue

    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]

    for c in cnts:
        if cv2.contourArea(c) < args["min_area"]
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        text = "Occupied"
    cv2.putText(frame, "Room Status: {}".format(text), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),(10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255),1)
    cv2.imshow("Security Fee", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()		
      
  
# free up memory
camera.release()
cv2.destroyAllWindows()
