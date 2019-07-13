# Tesseract4 (for Windows : https://github.com/UB-Mannheim/tesseract/wiki)
#            (for Linux   : sudo apt install tesseract-ocr libtesseract-dev)
# pip install pytesseract
# Usage : python3 ocr.py ocr1.png
import sys
import cv2
import pytesseract

filename = sys.argv[1]
image = cv2.imread(filename)
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] # threshold
#gray = cv2.medianBlur(gray, 3) # blur 

cv2.imshow('GRAY', gray)
text = pytesseract.image_to_string(gray)
print(text)

cv2.waitKey(0)
