# Install: pip install mtcnn
# Usage: python grab_face_nose_mouth.py Halsey.jpg
# adjust two pictures to have same size
from mtcnn import MTCNN
import cv2
import sys
import numpy as np

if len(sys.argv)>1:
    filename = sys.argv[1]
else:
    filename = 'Halsey.jpg'
	
img = cv2.imread(filename)
file = filename[:-4]
print(file)

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

detector = MTCNN()

faces = detector.detect_faces(rgb)

# only process for 1 face
for face in faces:
    # get face
    x,y,w,h = face['box'] # x,y,w,h
    print(x,y,w,h)    
    roi = img[y:y+h,x:x+w]
    cv2.imshow('face', roi)
    cv2.imwrite(file+'_face.jpg',roi)
    
	# draw keypoints	
    keypoints= face['keypoints']
    print(keypoints)
    
	# get coords
    left_eye_x,left_eye_y    = keypoints['left_eye']
    right_eye_x, right_eye_y = keypoints['right_eye']
    nose_x, nose_y = keypoints['nose']
    mouth_left_x, mouth_left_y   = keypoints['mouth_left']
    mouth_right_x, mouth_right_y = keypoints['mouth_right']
	
    # get nose 
    #eye_lo = min(left_eye_y, right_eye_y)	
    eye_hi = max(left_eye_y, right_eye_y)
    nose = img[ eye_hi+20: nose_y+20, nose_x-25 : nose_x+25] # for Halsey
    #nose = img[ eye_hi+30: nose_y+20, nose_x-25 : nose_x+30] # for Ariana
    print('nose:',nose.shape)
    cv2.imshow('nose', nose)
    cv2.imwrite(file+'_nose.jpg', nose)
	
	# get mouth
    mouth_y_lo = min(mouth_left_y, mouth_right_y)	
    mouth_y_hi = max(mouth_left_y, mouth_right_y)	
    mouth= img[ mouth_y_lo-10 : mouth_y_hi+25, mouth_left_x : mouth_right_x] 
    print('mouth:', mouth.shape)
    cv2.imshow('mouth', mouth)
    cv2.imwrite(file+'_mouth.jpg', mouth)
	
	# draw bbox & keypoints
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 2)
    img = cv2.circle(img, keypoints['left_eye'], 2, (255,255,255), 2)
    img = cv2.circle(img, keypoints['right_eye'], 2, (255,255,255), 2)	
    img = cv2.circle(img, keypoints['nose'], 2, (255,255,255), 2)	
    img = cv2.circle(img, keypoints['mouth_left'], 2, (255,255,255), 2)
    img = cv2.circle(img, keypoints['mouth_right'], 2, (255,255,255), 2)
	
cv2.imshow('MTCNN face detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
