# package import
import numpy as np
import matplotlib.pyplot as plt
import cv2
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Load Modules
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage
import scipy
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature

##########
# Generate noisy image of a square
image = np.zeros((384, 384), dtype=float)
image[96:-96, 96:-96] = 1

image = ndi.rotate(image, 15, mode='constant')
image = ndi.gaussian_filter(image, 4)
image = random_noise(image, mode='speckle', mean=0.1)
image = cv2.normalize(image,None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)

# Compute the Canny filter for two values of sigma
edges1 = feature.canny(image)
edges2 = feature.canny(image, sigma=3)

#########

image = cv2.imread(r'Images/goeppingen.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
image = cv2.imread(r'Images/road2.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
image = cv2.imread(r'Images/houses.bmp', cv2.IMREAD_GRAYSCALE).astype(np.float32)

Sobel = skimage.filters.sobel(image)
Sobel_N = cv2.normalize(Sobel,None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
thx = ((Sobel > 0.14)*255).astype(np.uint8)

#edges = (skimage.feature.canny(image, sigma=4, high_threshold=0.7, low_threshold = 0.5 , use_quantiles = True)*255).astype(np.uint8)
#edges = (skimage.feature.canny(image, sigma = 11,high_threshold=0.95, low_threshold = 0.8, use_quantiles = True)*255).astype(np.uint8)
edges = (skimage.feature.canny(image, sigma = 2, high_threshold=0.7, low_threshold = 0.4, use_quantiles = True)*255).astype(np.uint8)


cv2.imwrite(r'CannyExample_houses.jpg', edges)


'''
cv2.imwrite(r'CannyExample_Square.jpg', image)
cv2.imwrite(r'CannyExample_Square_Sobel.jpg', Sobel_N)
cv2.imwrite(r'CannyExample_Square_ThresholdEdges.jpg', thx)
cv2.imwrite(r'CannyExample_Square_CannyEdges.jpg', edges)
'''