# $pip install opencv-contrib-python

# webcam usage: python multi_object_tracking.py 
# mp4    usage: python multi_object_tracking.py --video horseracing.mp4 --tracker csrt
#               press s to select RoI, press c to cancel, press SPACE or ENTER to proceed
#               press q to quit
                     
# Tracker: csrt, kcf, boosting, mil, tld, medianflow, mosse 
# csrt  for slower FPS, higher object tracking accuracy
# kcf   for faster FPS, lower  object tracking accuracy
# mosse for fastest FPS

from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

# parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf", help="OpenCV object tracker type")
args = vars(ap.parse_args())

OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

trackers = cv2.MultiTracker_create()

# if no video path was not supplied, then get stream from Camera
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)
else:
	vs = cv2.VideoCapture(args["video"])


while True:
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame
 
	if frame is None: # reach end of frame
		break
 
	frame = imutils.resize(frame, width=600) # resize frame
	
	# grab bounding box corrdinates for each tracked objecct
	(success, boxes) = trackers.update(frame)

	for box in boxes:
		(x, y, w, h) = [int(v) for v in box]
		cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
			
	cv2.imshow("Frame", frame) # show output frame
	key = cv2.waitKey(1) & 0xFF

	# 's' key to select a bounding box to track
	if key == ord("s"):
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
		box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
 
		# create a new object tracker for the bounding box and add it
		# to our multi-object tracker
		tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
		trackers.add(tracker, frame, box)
		
	elif key == ord("q"): # 'q' key to quit the loop
		break

if not args.get("video", False): # if select webcam
	vs.stop()                    # then release the pointer
else:
	vs.release()                 # else release the file pointer
 
cv2.destroyAllWindows() # close all windows		
