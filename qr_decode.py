import cv2
import numpy as np
import sys
import time

if len(sys.argv)>1:
    inputImage = cv2.imread(sys.argv[1])
else:
    inputImage = cv2.imread("qrcode.jpg")

# Display barcode and QR code location
def display(img, bbox):
    n = len(bbox)
    for j in range(n):
        cv2.line(img, tuple(bbox[j][0]), tuple(bbox[(j+1)% n][0]), (255,0,0), 3)
    cv2.imshow("Result", img)

# QRCodeDetector 
qrDecoder = cv2.QRCodeDetector()

# Detect and decode the qrcode
data, bbox, rectifiedImage = qrDecoder.detectAndDecode(inputImage)
if len(data)>0:
    print("Decoded Data: {}".format(data))
    display(inputImage, bbox)
    rectifiedImage = np.uint8(rectifiedImage)
    cv2.imshow("Rectified QRcode", rectifiedImage)
else:
    print("QR Code not detected")
    cv2.imshow("Results", inputImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

