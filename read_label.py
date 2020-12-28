### read label file
# useage: python read_label.py 
#         python read_label.py 000000_0.png
#         python read_label.py ./Data_preprocessing/test_label 003218_0.png
import sys
import cv2

if len(sys.argv)<1:
    dir  = "."
    file = sys.argv[1]
else if len(sys.argv)==1:
    dir  = sys.argv[1]
    file = sys.argv[2]
else:
	dir  = "./Dataset/test_label"
	file = "000000_0.png"
	
image = cv2.imread(dir+"/"+file)
label = file.replace('.png','')
print(label)

# coords in y,x
coords = {
"000000_0": {'background':(27,27),'head':(26,90), 'neck':(60,85), 'upper-body':(89,103), 
             'right-arm':(120,58), 'left-arm':(129,131),'right-leg':(235,71), 'left-leg':(235,129)},
"000001_0": {'background':(27,27),'head':(31,113), 'neck':(73,108), 'upper-body':(136,94), 
             'right-arm':(213,47), 'left-arm':(228,133),'right-leg':(217,70), 'left-leg':(225,109)},
"000010_0": {'background':(27,27),'head':(31,113), 'neck':(73,108), 'upper-body':(136,108), 
             'right-arm':(119,60), 'left-arm':(240,136),'right-leg':(217,71), 'left-leg':(225,112)},
"000020_0": {'background':(27,27),'head':(37,102), 'neck':(68,88), 'upper-body':(111,84), 
'belly':(157,58),'right-arm':(147,39), 'left-arm':(168,126),'right-leg':(194,74), 'left-leg':(195,115)},
"003218_0": {'background':(27,27),'head':(26,90), 'neck':(60,85), 'upper-body':(89,103), 
             'right-arm':(120,58), 'left-arm':(129,131),'right-leg':(235,71), 'left-leg':(235,129)},
        }

try:
    print('background:',image[coords[label]['background']])
    print('head      :',image[coords[label]['head']]) 
    print('neck      :',image[coords[label]['neck']])  
    print('upper body:',image[coords[label]['upper-body']])
    print('right arm :',image[coords[label]['right-arm']])
    print('left  arm :',image[coords[label]['left-arm']])
    print('right leg :',image[coords[label]['right-leg']])
    print('left  leg :',image[coords[label]['left-leg']])
    print('belly     :',image[coords[label]['belly']])
	cv2.imshow('image',image)
except:
    print('input label number not in dictionary, need to be added')


cv2.waitKey(0)
cv2.destroyAllWindows()