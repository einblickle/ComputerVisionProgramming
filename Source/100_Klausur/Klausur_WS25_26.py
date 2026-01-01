import cv2 
import numpy as np
from skimage.morphology import skeletonize
from skimage import data
from skimage.util import invert
import skimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import modules.plotting as plotting

plt.close('all')
# load image
img = cv2.imread(r'images/coast1.bmp', cv2.IMREAD_ANYDEPTH)
img_landscape = cv2.imread(r'images/landscape.png', cv2.IMREAD_ANYDEPTH)
img_Malamute = cv2.imread(r'images/malamute.png', cv2.IMREAD_ANYDEPTH)
img_airfield = cv2.imread(r'images/airfield.tif', cv2.IMREAD_ANYDEPTH)
img_fruits = cv2.imread(r'./images/fruits.png', cv2.IMREAD_COLOR)
img_fruits = cv2.cvtColor(img_fruits, cv2.COLOR_BGR2RGB)



fig = plt.figure()
plt.imshow(img, cmap='gray')
plt.tick_params(axis='both', labelsize=25)
plt.show()


fig = plt.figure()
plt.imshow(img_fruits)
plt.tick_params(axis='both', labelsize=25)
plt.show()

plotting.displayImageAndAllHist(img, title = 'Bild')
plotting.displayImageAndAllHist(img_airfield, title = 'Bild')
plotting.displayImageAndAllHist(img_Malamute, title = 'Bild')
plotting.displayImageAndAllHist(img_landscape, title = 'Bild')

plotting.displayHistogram(img_landscape, title = 'Histogramm I')
plotting.displayHistogram(img_Malamute, title = 'Histogramm III')
plotting.displayHistogram(img, title = 'Histogramm II')
