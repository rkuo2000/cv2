# capture image from RPi3 usb camera (running mjpg_streamer), and display on PC
from PIL import Image
import urllib.request
import io
import numpy as np
import cv2

URL = 'http://192.168.43.176:8080/?action=snapshot -O rpi3_snapshot.jpg'

while 1:
    with urllib.request.urlopen(URL) as url:
        f=io.BytesIO(url.read())
    image = Image.open(f)
#    image.show()

    img = np.array(image)
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('CAM', frame)
    cv2.waitKey(1)

cv2.destroyAllWindows()
