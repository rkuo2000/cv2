# open browser at ipaddr of ESP32-CAM to set stream size
# 320x240 doesn't work, other resolution are OK
import numpy as np
import cv2
from urllib.request import urlopen

# port 81 has stream, see ESP32-CAM webserver.ino
url = 'http://192.168.1.5:81/stream' 
CAMERA_BUFFER_SIZE = 4096
stream = urlopen(url)
bbb=b''

while True:
    bbb += stream.read(CAMERA_BUFFER_SIZE)
    a = bbb.find(b'\xff\xd8')
    b = bbb.find(b'\xff\xd9')
    if a>-1 and b>-1:
        jpg = bbb[a:b+2]
        bbb = bbb[b+2:]
        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)
        cv2.imshow('CAM', img)
    cv2.waitKey(1)

cv2.destroyAllWindows()
