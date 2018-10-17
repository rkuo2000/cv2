import numpy as np
import cv2

img = cv2.imread('testpic.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
edge = cv2.Canny(blur, 100, 200)

cv2.imshow("edge", edge)

lines = cv2.HoughLinesP(edge, 1, np.pi/180, 100)

print(lines)

for i in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[i]:
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255), 2)

cv2.imshow("img", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

