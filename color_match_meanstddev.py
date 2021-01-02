## Color Matching from img2 to img1
# color matching good for nose, mouth, not good for face
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt

if len(sys.argv)>2:
    file1 = sys.argv[1] # src 
    file2 = sys.argv[2] # dst
else:
    file1 = 'Halsey_nose.jpg'
    file2 = 'Ariana_nose.jpg' 
	
reference = cv2.imread(file1)
print(reference.shape)
image = cv2.imread(file2)

# resize img2 to img1 
image = cv2.resize(image, (reference.shape[1], reference.shape[0])) # cv2.resize((h,w))
print(image.shape)

# need to conver to hsv (rgb won't work for facial color matching)
reference_hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# color matching
m1, sd1 = cv2.meanStdDev(reference_hsv)
m2, sd2 = cv2.meanStdDev(image_hsv)
m = m1-m2
print(m)
print(m.shape)
m = m.reshape(3).astype('uint8')

### match meanStdDev
#matched_hsv = image_hsv + m 

# found it has black dots, so use below check to avoid it
matched_hsv = image_hsv
for i in range(image_hsv.shape[0]):
	for j in range(image_hsv.shape[1]):
		if (image_hsv[i][j] + m).any() <=2 : matched_hsv[i][j] = image_hsv[i][j]
		else:                                matched_hsv[i][j] = image_hsv[i][j] + m

matched = cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite('matched.jpg', matched)

# display images
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)
for aa in (ax1, ax2, ax3):
    aa.set_axis_off()

ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax1.set_title('Source')
ax2.imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
ax2.set_title('Reference')
ax3.imshow(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB))
ax3.set_title('Matched')

plt.tight_layout()
plt.show()
