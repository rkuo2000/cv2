import numpy as np
import cv2
import sys

if len(sys.argv)>1:
    filename = sys.argv[1]
    x1 = sys.argv[2]
    y1 = sys.argv[3]
    x2 = sys.argv[4]
    y2 = sys.argv[5]
else:
    filename = "Hiwin-Robotic-Welding1.jpg"
    (x1,y1) = (444,84)
    (x2,y2) = (580,472)	
#    filename = "Hiwin-Writing-Robot.jpg"
#    (x1,y1) = (609,20)
#    (x2,y2) = (619,268)

img = cv2.imread(filename)
print(img[y1][x1]) # check color value
print(img[y2][x2]) # check color value

# Range of Color 
lower_red = np.array([0,0,220])
upper_red = np.array([50,50,255])
	
mask = cv2.inRange(img, lower_red, upper_red)
cv2.imshow('mask',mask)
cv2.imwrite(filename.replace('.jpg','_dot.jpg'), mask)

(cnts, hierarchy) = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

x = []
y = []
w = []
h = []
for i in range(len(cnts)):
    (X,Y,W,H) = cv2.boundingRect(cnts[i])
    print(X,Y,W,H) # print bounding box x,y,w,h
#    print(mask[Y][X]) # check color value
#    print(mask[Y+H-1][X+W-1]) # check color value
    x.append(X)
    y.append(Y)
    w.append(W)
    h.append(H)
x1 = x[0]+int(w[0]/2)
y1 = y[0]+int(h[0]/2)
x2 = x[1]+int(w[1]/2)
y2 = y[1]+int(h[1]/2)
print(abs(x2-x1), abs(y2-y1))

D = np.sqrt(abs(x1-x2)**2 +abs(y1-y2)**2)
print(D)

cv2.waitKey(0)
cv2.destroyAllWindows()
