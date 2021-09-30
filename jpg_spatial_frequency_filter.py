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

#img_freq = np.fft.fft2(raw_img) # convert to frequency domain
#img_amp  = np.fft.fftshift(np.abs(img_freq)) # calculate amplitude spectrum

lp_filt = psyfilter.butter2d_lp(size=raw_img.shape, cutoff=0.05, n=10)
lp_filt_disp = lp_filt * 2.0 - 1.0 # convert it to -1:1 for display

plt.imshow(lp_filt_disp, cmap='gray')
plt.axis('off')
plt.show()
