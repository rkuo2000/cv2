import cv2

camera = cv2.VideoCapture(0)
subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=True)

while 1:
  ret, frame = camera.read()
  
  mask = subtractor.apply(frame) 
  res = cv2.bitwise_and(frame, frame, mask=mask)
  
#Gaussian blur & Threshold
  #Convert image to gray scale
  gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
  cv2.imshow('gray',gray)
  
  #By Gaussian blurring, we create smooth transition from one color to another and reduce the edge content
  blur = cv2.GaussianBlur(gray, (5, 5), 0)
  cv2.imshow('blur',blur)
  
  #use thresholding to create binary images from grayscale images
  ret, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
  cv2.imshow('thresh', thresh) 
  
#Contour & Hull & Convexity
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   
  
  hull = cv2.convexHull(res)
  cv2.imshow('hull', hull) 
  
  defects = cv2.convexityDefects(res, hull)
  cv2.imshow('detect', detect)   

  cv2.waitKey(1)
  
# free up memory
camera.release()
cv2.destroyAllWindows()
