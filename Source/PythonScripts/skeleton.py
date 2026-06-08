# Import the necessary libraries

###
## https://gist.github.com/neeru1207/dc30df52237d5c58ded47c43ed3dcf89
###

import cv2
import numpy as np

# Read the image as a grayscale image
from skimage.morphology import skeletonize
from skimage import data
from skimage.util import invert
import skimage

img = 255 - cv2.imread(r'.\img\cat.png', cv2.IMREAD_ANYDEPTH)

'''
img = cv2.imread('A://testimg5.jpg', 0)
img = invert(data.horse()).astype('uint8')
'''
img = img.astype('uint8')*255
img_orig = img.copy()


# Step 1: Create an empty skeleton
size = np.size(img)
skel = np.zeros(img.shape, np.uint8)

# Get a Cross Shaped Kernel
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

# Repeat steps 2-4
while True:
    #Step 2: Open the image
    Mopen = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
    #Step 3: Substract open from the original image
    temp = cv2.subtract(img, Mopen)
    #Step 4: Erode the original image and refine the skeleton
    eroded = cv2.erode(img, element)
    skel = cv2.bitwise_or(skel,temp)
    img = eroded.copy()
    # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
    if cv2.countNonZero(img)==0:
        break


skeleton_SK = skimage.morphology.skeletonize(img_orig).astype('uint8')*255
skel = skel.astype('uint8')*255

cv2.imwrite(r'.\img\cat_skeleton.png', skeleton_SK)
cv2.imwrite(r'.\img\cat_skeleton_simple.png',skel)