import numpy as np
import cv2

cap = cv2.VideoCapture(0)
 
while(1):
 ret, img = cap.read() 
 
 height, width, channel = img.shape
 mask = np.zeros(img.shape[:2],np.uint8)
  
 bgdModel = np.zeros((1,65),np.float64)
 fgdModel = np.zeros((1,65),np.float64)
  
 rect = (0,0,height,width)
 cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
  
 mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
 img = img*mask2[:,:,np.newaxis]
  
 cv2.imshow('CAM', img) 
 ch = cv2.waitKey(0)
 if ch==27: #wait for ESC key to exit
    break
    
cap.release()
cv2.destroyAllWindows()
                   
