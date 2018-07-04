import urllib.request
import numpy as np
import cv2

URL = 'https://cdn.omlet.co.uk/images/originals/rabbit-health-symptoms.jpg'

while 1:
    resp = urllib.request.urlopen(URL)
    img_array = np.array(bytearray(resp.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, -1)

    cv2.imshow('CAM', img)
    cv2.waitKey(1)

cv2.destroyAllWindows()
