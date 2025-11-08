# package import
import numpy as np
import matplotlib.pyplot as plt
import cv2
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules import plotting
# Load Modules
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage
import scipy

# Bilddatei einladen
img_air = cv2.imread(r'images/airfield02g.png', cv2.IMREAD_GRAYSCALE)
img_bricks = cv2.imread(r'images/bricks.jpg', cv2.IMREAD_GRAYSCALE)
img_test = cv2.imread(r'images/testpattern.bmp', cv2.IMREAD_GRAYSCALE)
img_road = cv2.imread(r'images/road2.jpg', cv2.IMREAD_GRAYSCALE)
img_houses = cv2.imread(r'images/houses.bmp', cv2.IMREAD_GRAYSCALE)
# select image to analyze

image = img_houses
image_f = image.astype('float32')

## Blurring / Gaussian Blur / Flat Top

kernelsize = 5
kernel = np.ones((kernelsize, kernelsize), np.float64) / (kernelsize * kernelsize)
blur = cv2.filter2D(image_f, -1, kernel)

# gaussian blur
kernel = (5,5)
sigma = 2
gaussian = cv2.GaussianBlur(image, kernel, sigma)

#several blur steps in a row
kernel = (3,3)
noOfSteps = 100
blurSteps = image.copy()
for i in range(noOfSteps):
    blurSteps = cv2.blur(blurSteps, kernel)

# Glättung vor der Kantenfilterung
image_smoothed = cv2.GaussianBlur(image_f, (11,11), 5)
image_smoothed = image_f


kernelSobelX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
sobelX = cv2.filter2D(image_smoothed, -1, kernelSobelX)


kernelSobelY = np.array([[-1,-2,-1],[0,0,0],[1,2,1]],np.float32)
sobelY = cv2.filter2D(image_smoothed, -1, kernelSobelY)

# combine the directions (Pythagoras)
edges_sobel = np.sqrt(sobelX**2 + sobelY**2)

# scale the image back to uint8
sobelX =  cv2.normalize(sobelX, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
sobelY =  cv2.normalize(sobelY, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
edges_sobel =  cv2.normalize(edges_sobel, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')


#Filter Testen


kernel = np.array([[-1,-1,-1],[-1,12,-1],[-1,-1,-1]]).astype('float32')
kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]).astype('float32')
#kernel = np.ones((3,3)).astype('float32')
#kernel = (1/np.sum(kernel))*kernel


# apply
filtered = cv2.filter2D(image_f, -1, kernel)
#filtered = cv2.normalize(np.abs(filtered), None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
#filtered = np.clip(filtered,0, 255). astype('uint8')
#filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
SKLaplace = skimage.filters.laplace(image_f, 3)
