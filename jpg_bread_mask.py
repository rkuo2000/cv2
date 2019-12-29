import numpy as np
import cv2

DIR_PATH   = './bread'
BREAD_NAME = 'croissant'
BREAD_PATH= './bread/'+BREAD_NAME

for i in range(0,25):
    file = BREAD_PATH+'-'+str(i)
    print(file+'.jpg')	
    img = cv2.imread(file+'.jpg')

    # Range of Color
    lower_white = np.array([200,200,200])
    upper_white = np.array([255,255,255])

    mask = cv2.inRange(img, lower_white, upper_white)
    mask = cv2.bitwise_not(mask)

    x, y, w, h = cv2.boundingRect(mask)
    #bbox = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
    #cv2.imshow('image', img)

    img = img[y:y+h, x:x+w]
    mask = cv2.inRange(img, lower_white, upper_white)
    mask = cv2.bitwise_not(mask)
    cv2.imshow('mask', mask)
    cv2.imwrite(file+'-mask.jpg', mask)

    res = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('result', res)
    cv2.imwrite(file+'.jpg', res)

    with open(BREAD_PATH+str(i)+'.xml', "w") as text_file:
        text_file.write("            <object>\n")	
        text_file.write("                    <name>croissant%s</name>\n" % str(i))
        text_file.write("                    <pose>Frontal</pose>\n")	
        text_file.write("                    <truncated>0</truncated>\n")	
        text_file.write("                    <difficult>0</difficult>\n")	
        text_file.write("                    <occluded>0</occluded>\n")
        text_file.write("                    <bndbox>\n")
        text_file.write("                            <xmin>%s</xmin>\n" % str(x))
        text_file.write("                            <xmax>%s</xmax>\n" % str(x+w))
        text_file.write("                            <ymin>%s</ymin>\n" % str(y))
        text_file.write("                            <ymax>%s</ymax>\n" % str(y+h))	
        text_file.write("                    </bndbox>\n")
        text_file.write("            </object>\n")
	
cv2.waitKey(0)
cv2.destroyAllWindows()

