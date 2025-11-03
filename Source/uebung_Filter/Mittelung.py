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

testImg  = np.zeros((600,600)).astype('uint8')

#circles
cv2.circle(testImg, (100, 150), 50, 255, -1, cv2.FILLED)
cv2.circle(testImg, (250, 150), 6, 255, -1, cv2.FILLED)
cv2.circle(testImg, (400, 150), 20, 255, -1, cv2.FILLED)

#points
testImg[150, 200] = 255
testImg[150, 325] = 255

#lines
cv2.line(testImg, (100, 310), (100, 390), 255, 4, cv2.LINE_8)
cv2.line(testImg, (60, 350), (140, 350), 255, 4, cv2.LINE_8)

cv2.line(testImg, (200, 410), (200, 490), 255, 1, cv2.LINE_8)
cv2.line(testImg, (160, 450), (240, 450), 255, 1, cv2.LINE_8)

cv2.line(testImg, (370, 320), (430, 370), 255, 1, cv2.LINE_8)
cv2.line(testImg, (370, 370), (430, 320), 255, 1, cv2.LINE_8)

cv2.line(testImg, (470, 320), (530, 370), 255, 2, cv2.LINE_8)
cv2.line(testImg, (470, 370), (530, 320), 255, 2, cv2.LINE_8)

cv2.line(testImg, (370, 470), (430, 520), 255, 3, cv2.LINE_8)
cv2.line(testImg, (370, 520), (430, 470), 255, 3, cv2.LINE_8)






# Bilddatei einladen
img_air = cv2.imread(r'images/airfield02g.png', cv2.IMREAD_GRAYSCALE)
img_bricks = cv2.imread(r'images/bricks.jpg', cv2.IMREAD_GRAYSCALE)
img_test = cv2.imread(r'images/testpattern.bmp', cv2.IMREAD_GRAYSCALE)
img_road = cv2.imread(r'images/road2.jpg', cv2.IMREAD_GRAYSCALE)

# select image to analyze

image = img_test
image_f = image.astype('float64')

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


kernelSobelX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float64)
sobelX = cv2.filter2D(image_smoothed, -1, kernelSobelX)


kernelSobelY = np.array([[-1,-2,-1],[0,0,0],[1,2,1]],np.float64)
sobelY = cv2.filter2D(image_smoothed, -1, kernelSobelY)

# combine the directions (Pythagoras)
edges_sobel = np.sqrt(sobelX**2 + sobelY**2)

# scale the image back to uint8
sobelX =  cv2.normalize(sobelX, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
sobelY =  cv2.normalize(sobelY, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
edges_sobel =  cv2.normalize(edges_sobel, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')


#Bild Schärfen:


kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]]).astype('float64')


# apply
laplace = cv2.filter2D(image_smoothed, -1, kernel)
laplace_skimage = skimage.filters.laplace(image_smoothed, ksize=5)
laplace_OpenCV = cv2.Laplacian(image_smoothed, cv2.CV_64F, 3)
# scale the image back to uint8
laplace2 = cv2.normalize(np.abs(laplace), None, 0, 255, cv2.NORM_MINMAX).astype('uint8')


