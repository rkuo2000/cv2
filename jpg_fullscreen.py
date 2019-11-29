import cv2

image = cv2.imread('fullscreen.jpg')

top, bottom, left, right = 20, 244, 208, 432 # 224x224
roi = image[top:bottom, left:right]          # region of interest
        	
cv2.rectangle(image, (left, top), (right, bottom), (255,0,0), 2)

text = "Hello, How are you ?"
cv2.putText(image, text, (left, bottom+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2) 

cv2.namedWindow('fullscreen', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('fullscreen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow('fullscreen',image)

cv2.waitKey(0)
cv2.destroyAllWindows()
