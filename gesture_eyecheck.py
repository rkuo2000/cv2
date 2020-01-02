import cv2
import sys
import os
import time
from random import *
import numpy as np
from gtts import gTTS
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model

camera = cv2.VideoCapture(0)

# check camera resolution
(_, frame) = camera.read()
(height, width, channel) = frame.shape
print(width, height)

# Load Model
model = load_model('model/gesture_cnn.h5')

# Dictionary
dict = {0: 'down', 1: 'left', 2: 'right', 3: 'stop', 4: 'up'}

# generate voice files
#tts = gTTS('剪刀,石頭,布', lang='zh-tw')
#tts.save('paper-rock-scissors.mp3')

# read images
img = cv2.imread('eyecheck.jpg')
rows,cols = img.shape[:2]

# score
score_ok = 0
score_fail = 0

# Up image
img0 = img # Up

# Left image
M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
img1 = cv2.warpAffine(img, M, (cols, rows))

# Down image
M = cv2.getRotationMatrix2D((cols/2, rows/2), 180, 1)
img2 = cv2.warpAffine(img, M, (cols, rows))

# Right image
M = cv2.getRotationMatrix2D((cols/2, rows/2), 270, 1)
img3 = cv2.warpAffine(img, M, (cols, rows))

# computer roi
x = randint(0, 4)
if x==0:
    eyecheck = img0
    ans = 'up'
if x==1:
    eyecheck = img1
    ans = 'left'
if x==2:
    eyecheck = img2
    ans = 'down'
if x==3:
    eyecheck = img3
    ans = 'right'
	
while True:
    (_, frame) = camera.read()
    frame = cv2.flip(frame, 1) # flip the frame so that it is not the mirror view
	
    top, bottom, left, right = 50, 50+120, 50, 50+120 # 224x224
    frame[top:bottom, left:right] = eyecheck	
		
    top, bottom, left, right = 20, 244, 308, 532 # 224x224
    roi = frame[top:bottom, left:right]          # region of interest
        	
    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
	
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_bgr  = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    frame[top:bottom, left:right] = roi_bgr

    res = cv2.resize(roi_bgr, (96, 96), interpolation = cv2.INTER_CUBIC)	
    x_data = res / 255.0    
    x_data = x_data.reshape(1,96,96,3)
    
    # prediction
    predictions = model.predict(x_data)
    maxindex = int(np.argmax(predictions))
    print(predictions[0][maxindex], dict[maxindex])
	
    cv2.putText(frame, dict[maxindex], (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
    cv2.putText(frame, str(predictions[0][maxindex]), (left, bottom+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)       
    if (dict[maxindex]==ans):
        cv2.putText(frame, 'PASS', (320, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2) 
    else:
        cv2.putText(frame, 'FAIL', (320, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2) 	
    cv2.imshow('frame', frame)

    keypress = cv2.waitKey(1) & 0xFF # keypress by user
    if keypress == ord("q"):         # press q to play
        break
    if keypress == ord(" "):         # press q to play
        x = randint(0, 4)
        if x==0:
            eyecheck = img0
            ans = 'up'
        if x==1:
            eyecheck = img1
            ans = 'left'
        if x==2:
            eyecheck = img2
            ans = 'down'
        if x==3:
            eyecheck = img3
            ans = 'right'
		   
# free up memory
cv2.waitKey(0)
camera.release()
cv2.destroyAllWindows()

