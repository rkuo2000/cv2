# pip install opencv-python==4.5.3.56
import cv2
import numpy as np
from psychopy.visual import filters

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 0)
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raw_img = (gray /255.0) *2.0 - 1.0 # convert to -1:+1 range
    rms = 0.2

    raw_img = raw_img - np.mean(raw_img)  # make mean to be zero
    raw_img = raw_img / np.std(raw_img)   # make standard deviation to be 1
    raw_img = raw_img * rms

    img_freq = np.fft.fft2(raw_img) # convert to frequency domain
    img_amp  = np.fft.fftshift(np.abs(img_freq)) # calculate amplitude spectrum

    lp_filt = filters.butter2d_lp(size=raw_img.shape, cutoff=0.05, n=10)

    # applying spatial frequency filter
    img_filt = np.fft.fftshift(img_freq) * lp_filt
    img_filt_amp = np.abs(img_filt)

    # for display, take the logarithm
    img_filt_amp_disp = np.log(img_filt_amp + 0.0001)

    img_filt_amp_disp = (((img_filt_amp_disp - np.min(img_filt_amp_disp)) * 2) / np.ptp(img_filt_amp_disp)) -1

    cv2.imshow("orignal", gray)
    cv2.imshow("spatial", img_filt_amp_disp)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
