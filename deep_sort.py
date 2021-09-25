# pip install deep-sort-realtime
import sys
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=30, nn_budget=70, override_track_class=None)

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
    #frame = cv2.flip(frame, 0) # vertical flip (for opencv-python 4.5.3.56)
    #frame = cv2.flip(frame, 1) # horizontal flip (for opencv-python 4.5.3.56)
    print(frame.shape) 	#no need to flip for opencv-python 4.2.0.32)
    bbs = object_detector.detect(frame)
    tracks = trackers.update_tracks(bbs, frame=frame)
    for track in tracks:
        track_id = track.track_id
        ltrb = track.to_ltrb()
   
    cv2.imshow('CAM', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
