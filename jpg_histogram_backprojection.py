import numpy as np
import cv2

src = cv2.imread('cows.jpg')
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

ch = (0, 0)
hue = np.empty(hsv.shape, hsv.dtype)
cv2.mixChannels([hsv], [hue], ch)
window_image = 'Source image'
cv2.namedWindow(window_image)

def Hist_and_Backproj(val):
    bins = val
    histSize = max(bins, 2)
    ranges = [0, 180] # hue_range
    
    hist = cv2.calcHist([hue], [0], None, [histSize], ranges, accumulate=False)
    cv2.normalize(hist, hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    backproj = cv2.calcBackProject([hue], [0], hist, ranges, scale=1) 
    cv2.imshow('BackProj', backproj)
    
    w = 400
    h = 400
    bin_w = int(round(w / histSize))
    histImg = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(bins):
        cv2.rectangle(histImg, (i*bin_w, h), ( (i+1)*bin_w, h - int(np.round( hist[i]*h/255.0 )) ), (0, 0, 255), cv2.FILLED)
		
    cv2.imshow('Histogram', histImg)

bins = 25
cv2.createTrackbar('* Hue  bins: ', window_image, bins, 180, Hist_and_Backproj )
Hist_and_Backproj(bins)
cv2.imshow('window_image', src)
	
cv2.waitKey(0)
cv2.destroyAllWindows()
