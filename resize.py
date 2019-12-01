# resize images in a folder
import cv2
import sys
import os

path = sys.argv[1]
files = os.listdir(path)
print(files)

for file in files:
    img = cv2.imread(path+'/'+file)
    res = cv2.resize(img, (96,96), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(path+'/'+file, res)
