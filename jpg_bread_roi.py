import numpy as np
import cv2

BREAD_NAME = 'croissant'
BREAD_PATH= './'+BREAD_NAME+'/'

for i in range(0,20):
    file = BREAD_PATH+BREAD_NAME+'-'+str(i)
    print(file+'.jpg')	
    img = cv2.imread(file+'.jpg')

    # Range of Color
    lower_white = np.array([240,240,240])
    upper_white = np.array([255,255,255])
	
    mask = cv2.inRange(img, lower_white, upper_white)
    mask = cv2.bitwise_not(mask)

    x, y, w, h = cv2.boundingRect(mask)
    print(x,y,w,h)
    #bbox = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
    #cv2.imshow('image', img)

    roi = img[y:y+h, x:x+w]
    cv2.imshow('result', roi)
    cv2.imwrite(file+'-roi.jpg', roi)
	
cv2.waitKey(0)
cv2.destroyAllWindows()

