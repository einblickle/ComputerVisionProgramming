import numpy as np
import matplotlib.pyplot as plt
import scipy


noOfDataPoints = 5000
noOfOscillations = 10
amplitude = 10

# Create a sin shaped signal
xValues = np.linspace(0, 2*np.pi*noOfOscillations, noOfDataPoints )
yValues = amplitude * np.cos(xValues)

# add noise to the signal

noiseAmplitude = 3

randomNoise = noiseAmplitude * np.random.rand(noOfDataPoints)
yValues_noisy = yValues + randomNoise

# filter the signal
filterSize = 50
kernel = 1/filterSize * np.ones(filterSize)
filtered = scipy.signal.convolve(yValues_noisy, kernel, mode='same')

plt.figure()
plt.plot(yValues_noisy, 'r')
plt.plot(filtered, 'b')