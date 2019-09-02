import numpy as np

from matplotlib import pyplot as plt

import cv2

import scipy.fftpack

# images imput, to gray scale

img1 = cv2.imread('Moritz.jpg')
gray=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

#cv2.imshow('Original image',img1)
#cv2.imshow('Grayscale',gray)

# Images Input, layout, and transforms

f = np.fft.fft2(gray) #compute 2 dimensional discrete fourier transform
fshift = np.fft.fftshift(f) #shift the zero-freq components to the center of the spectrum
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img1, cmap = 'gray')
plt.title('Input Spectrum'), plt.xticks([]), plt.yticks([])

#plt.subplot(121),plt.imshow(gray, cmap = 'gray') #img a gray
#plt.title('Grayscale Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.show()
cv2.waitKey()


