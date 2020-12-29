# openCV color space conversion from BGR to RGB
import sys
import cv2

if len(sys.argv)>1:
    filename = sys.argv[1]
else:
    filename = "test.jpg"
	
img = cv2.imread(filename)
rgb= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imshow('BGR',img)
cv2.imshow('RGB',rgb)

cv2.imwrite(filename.replace('.jpg','_bgr2rgb.jpg'), rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()
