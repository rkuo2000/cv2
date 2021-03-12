import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

if len(sys.argv)>1:
    img = cv2.imread(sys.argv[1])
    X1 = int(sys.argv[2])
    Y1 = int(sys.argv[3])
    X2 = int(sys.argv[4])
    Y2 = int(sys.argv[5])
else:
    img = cv2.imread('messi5.jpg')
    X1 = 50
    Y1 = 50
    X2 = 450
    Y2 = 290
	
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (X1,Y1,X2,Y2)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

#plt.imshow(img),plt.colorbar(),plt.show()
cv2.imshow('grabCut', img)
cv2.imwrite('res.jpg',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
