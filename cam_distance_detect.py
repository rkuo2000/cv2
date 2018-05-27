import cv2
import numpy as np
import time
import math as m
from numpy import*

# assign color
g_min = np.array([30,70,30])
g_max = np.array([70,255,255])

k = np.ones((5,5), np.uint8)

def color_track(im, h_min, h_max):
    im_h = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(im_h, h_min, h_max,)
    #mask = cv2.medianBlur(mask, 7)
    #mask = cv2.dilate(mask, k, iterations=2)
    #im_c = cv2.bitwise_and(im, im, mask=mask)
    return mask #,im_c

def index_emax(cnt):
    max_num = 0
    max_i   =-1
    for i in range(len(cnt)):
        cnt_num=len(cnt[i])
        if cnt_num > max_num:
            max_num = cnt_num
            max_i   = i
    return max_i

def main():
    h1 = 175
    L1 = 700
    H  = 167 #green
    Z  = 0

    cap = cv2.VideoCapture(0)
    while True:
        im = cap.read()[1]
        mask1 = color_track(im,g_min,g_max)
        mask2 = color_track(im,g_min,g_max)
        # Red contour
        cnt = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        n = index_emax(cnt)
        if n != -1:
            hull = cv2.convexHull(cnt[n])
            cv2.drawContours(im,[hull],0,(0,200,0),3)
            M1 = cv2.moments(cnt[n])
            cx,cy = int(M1["m10"]/M1["m00"]), int(M1["m01"]/M1["m00"])
        else:
            cx, cy = 320, 240

        # Red region estimate
        mask = cv2.bitwise_and(mask1,mask1,mask=mask2)
        mask = cv2.Canny(mask2,100,200)
        y,x = np.where(mask==255)
        if len(y)!=0:
            ymax,ymin= np.amax(y),np.amin(y)
            h2 = ymax - ymin
            L2 = (h1/float(h2))*L1
            a  =H/float(h2)
            X = (cx-320)*a
            Y = (ymax-cy)*a
            if L2>X:
                Z=sqrt(L2*L2-X*X)
            X,Y,Z,L2 = round(X),round(Y),round(Z),round(L2)
            # Display Result
            cv2.circle(im,(cx,cy),5,(0,0,255), -1)
            cv2.putText(im,"X:"+str(X)+"[mm]",(30,20),1,1.5,(70,70,220),2)
            cv2.putText(im,"Y:"+str(Y)+"[mm]",(30,50),1,1.5,(70,70,220),2)
            cv2.putText(im,"Z:"+str(Z)+"[mm]",(30,80),1,1.5,(70,70,220),2)
            cv2.putText(im,"h2:"+str(h2)+"[pixcel]",(30,120),1,1.5,(220,70,90),2)
            cv2.putText(im,"L2:"+str(L2)+"[mm]",(30,160),1,1.5,(220,70,90),2)
            cv2.imshow("Camera",im)
            cv2.imshow("Mask",mask2)
        if cv2.waitKey(10)>0:
            cap.release()
            cv2.destroyAllWindows()
            break
if __name__ == '__main__':
    main()

        
