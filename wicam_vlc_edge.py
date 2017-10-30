# Install VLC Player on PC
# Add Environtment System Variables: VLC_PLUGIN_PATH = C:\Program Files\VideoLAN\VLC\plugins
# pip install python-vlc
# WiFi connected to WiCam module (streaming video)
import cv2
import vlc

#player=vlc.MediaPlayer('rtsp://192.168.100.1/cam1/h264')
player=vlc.MediaPlayer('rtsp://192.168.100.1/cam1/mpeg4')
player.play()

while 1:
    player.video_take_snapshot(0,'c:/wicam.png',0, 0)
    img = cv2.imread('c:/wicam.png')
    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,100,200)
    cv2.imshow('Edges', edges)
    cv2.waitKey(1)

cv2.destroyAllWindows()
