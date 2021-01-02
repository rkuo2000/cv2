## Color Matching from img2 to img1
# color matching good for nose, mouth, not good for face
import numpy as np
import sys

from skimage import data
from skimage import color
from skimage.io import imread, imsave
from skimage import exposure
from skimage.transform import resize, match_histograms
import matplotlib.pyplot as plt

if len(sys.argv)>2:
    file1 = sys.argv[1] # src 
    file2 = sys.argv[2] # dst
else:
    file1 = 'Halsey_nose.jpg'
    file2 = 'Ariana_nose.jpg' 
	
reference = imread(file1)
print(reference.shape)
image = imread(file2)

# resize img2 to img1 
image = resize(image, (reference.shape[0], reference.shape[1]))
print(image.shape)

# need to conver to hsv (rgb won't work for facial color matching)
reference_hsv = color.rgb2hsv(reference)
image_hsv = color.rgb2hsv(image)

# color matching
matched_hsv = match_histograms(image_hsv, reference_hsv, multichannel=True)
matched = color.hsv2rgb(matched_hsv)
imsave('matched.jpg', matched)

# show matched 
matched = plt.imread('matched.jpg')
#plt.imshow(matched)
#plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)
for aa in (ax1, ax2, ax3):
    aa.set_axis_off()

ax1.imshow(image)
ax1.set_title('Source')
ax2.imshow(reference)
ax2.set_title('Reference')
ax3.imshow(matched)
ax3.set_title('Matched')

plt.tight_layout()
plt.show()