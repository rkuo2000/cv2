import numpy as np
import cv2

img = cv2.imread('testpic.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
edge = cv2.Canny(blur, 100, 200)

cv2.imshow("edge", edge)

lines = cv2.HoughLines(edge, 1, np.pi/180, 100)

print(lines)

for i in range(0, len(lines)):
    rho=lines[i][0][0]
    theta=lines[i][0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0+1000*(-b))
    y1 = int(y0+1000*( a))
    x2 = int(x0-1000*(-b))
    y2 = int(y0-1000*( a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255), 1)

cv2.imshow("img", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

