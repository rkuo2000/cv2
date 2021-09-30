import cv2
import numpy as np
from matplotlib import pyplot as plt
import psychopy.filters as psyfilter

raw_img = cv2.imread('unsw_bw.jpg',0)

raw_img = (raw_img /255.0) *2.0 - 1.0 # convert to -1:+1 range
rms = 0.2

raw_img = raw_img - np.mean(raw_img)  # make mean to be zero
raw_img = raw_img / np.std(raw_img)   # make standard deviation to be 1
raw_img = raw_img * rms

img_freq = np.fft.fft2(raw_img) # convert to frequency domain
img_amp  = np.fft.fftshift(np.abs(img_freq)) # calculate amplitude spectrum

lp_filt = psyfilter.butter2d_lp(size=raw_img.shape, cutoff=0.05, n=10)

# applying spatial frequency filter
img_filt = np.fft.fftshift(img_freq) * lp_filt
# conver back to an image
img_new = np.real(np.fft.ifft2(np.fft.ifftshift(img_filt)))

# convert to mean zero and specified RMS contrast
img_new = img_new - np.mean(img_new)
img_new = img_new / np.std(img_new)
img_new = img_new * rms
img_new = np.clip(img_new, a_min=-1.0, a_max=1.0) 

plt.imshow(img_new, cmap='gray')
plt.axis('off')
plt.show()
