# pip install ultralytics

import os
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    #frame = cv2.flip(frame, 1) # mirror

    ## Object Detection 
    results = model(frame)

    ## Object Counting
    labels = results[0].names
    cls = results[0].boxes.cls.tolist()
    unique = list(dict.fromkeys(cls))

    sl = "en"
    text = "There are "
    for label in unique:
        count = cls.count(label)
        text = text + str(count) + " " + labels[int(label)] + ","
    print(text)

    results[0].save("out.jpg")
    img = cv2.imread("out.jpg")
    cv2.imshow('webcam', img)

    k = cv2.waitKey(1) & 0xFF
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()
