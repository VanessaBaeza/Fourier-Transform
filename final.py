import sys
sys.path.append('/home/vanessa/.local/lib/python2.7/site-packages') # point to your tensorflow dir
sys.path.append('home/vanessa/models/research/slim') # point ot your slim dir

import numpy as np

from matplotlib import pyplot as plt

import cv2

import scipy.fftpack

import numpy as np
import cv2
import random
import python_utils
#from python_utils import CFEVideoConf, image_resize
import glob
import math

# images imput, to gray scale

cap = cv2.VideoCapture(0)
frames_per_second=20
#config = CFEVideoConf(cap, filepath=save_path, res='480p')

def grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_fourier(gray):
    f = np.fft.fft2(gray) #compute 2 dimensional discrete fourier transform
    fshift = np.fft.fftshift(f) #shift the zero-freq components to the center of the spectrum
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    return plt.imshow(magnitude_spectrum, cmap = 'gray')

##def grayscale(frame):
##    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
##    f = np.fft.fft2(gray) #compute 2 dimensional discrete fourier transform
##    fshift = np.fft.fftshift(f) #shift the zero-freq components to the center of the spectrum
##    magnitude_spectrum = 20*np.log(np.abs(fshift))
##    return magnitude_spectrum


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Capture frame-by-frame
    ret, image = cap.read()
    img = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)

    img_and_magnitude = np.concatenate((img, magnitude_spectrum), axis=1)
    cv2.imshow('imagen',img_and_magnitude)


    gray=grayscale(frame)
    cv2.imshow('grayscale', gray)

    #plt.subplot(122),plt.imshow(grayscale(frame), cmap = 'gray')
    #plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    #fourier=apply_fourier(gray)
    #cv2.imshow('Fourier',fourier1)

    #fourier=apply_fourier(gray)

    #cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
