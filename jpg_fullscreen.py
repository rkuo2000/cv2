import cv2
import sys

if (len(sys.argv)>1):
    image = cv2.imread(sys.argv[1])
else:
    image = cv2.imread('wallpaper.jpg')

height, width, channel = image.shape

screen_width =1920
screen_height=1080
if (width!=screen_width and height!=screen_height):
    img=cv2.resize(image, (screen_width, screen_height), interpolation = cv2.INTER_CUBIC)

# put text 
text = "Wallpaer: Width= "+str(width)+" Height= "+str(height)
x = int(screen_width/4)
y = int(screen_height/4)
cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2) 

# put text 
text = "Screen: Width= "+str(screen_width)+" Height= "+str(screen_height)
x = int(screen_width/4)
y = int(screen_height/2)
cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2) 

# Rectangle area
top, bottom, left, right = 20, 244, 208, 432 # 224x224
cv2.rectangle(img, (left, top), (right, bottom), (255,0,0), 2)

text = "putText demo"
cv2.putText(img, text, (left, bottom+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2) 

# region of interest
roi = img[top:bottom, left:right]

cv2.namedWindow('fullscreen', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('fullscreen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow('fullscreen',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
