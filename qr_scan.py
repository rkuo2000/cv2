# Usage: python qr_scan.py -i qr_csu.jpg
from pyzbar import pyzbar
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
barcodes = pyzbar.decode(image)

for barcode in barcodes:
    (x,y,w,h)=barcode.rect
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
    barcodeData = barcode.data.decode("utf-8")
    barcodeType = barcode.type

    text = "{} ({})".format(barcodeData, barcodeType)
    cv2.putText(image, text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
