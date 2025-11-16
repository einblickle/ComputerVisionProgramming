import numpy as np
import cv2
import plotly.express as px
import plotly.graph_objects as go
import skimage



center = (1686, 1500)
radius = 1300

image = cv2.imread(r'registration_full.tiff', cv2.IMREAD_ANYDEPTH)





mask = np.zeros(image.shape)
mask = cv2.circle(mask, center = center, radius = radius, color = (255,255,255), thickness = -1)
mask = cv2.blur(mask, (101, 101))
mask = cv2.blur(mask, (101, 101))
mask = cv2.blur(mask, (101, 101))
mask = cv2.blur(mask, (101, 101))

mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX).astype('float')

maskedImage = image * mask
maskedImage = cv2.normalize(maskedImage, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

# define search radius for maximum and minimum filter
disk41 = skimage.morphology.disk(41)
disk11 = skimage.morphology.disk(11)

min_image = skimage.filters.rank.minimum(maskedImage, disk41)
max_image = skimage.filters.rank.maximum(maskedImage, disk41)

max_brightnessMap = np.max(max_image)/(max_image.astype(np.float)+1)

bkg = skimage.filters.rank.maximum(min_image, disk51)



bkg_corr = maskedImage-min_image

canny_filtered = skimage.feature.canny(maskedImage, sigma=2)

dilated = skimage.morphology.binary_dilation(canny_filtered, np.ones((5,5)))
eroded = skimage.morphology.binary_erosion(dilated, np.ones((5,5)))

skeleton = skimage.morphology.skeletonize(eroded)


'''

min_image_b = cv2.blur(min_image, (21,21))
#bkg_corr_blur = cv2.GaussianBlur(bkg_corr,(9,9), 3)

bkg_corr_blur = cv2.medianBlur(bkg_corr, 3)

laplace_filtered = skimage.filters.laplace(bkg_corr_blur, ksize=7)

canny_filtered = skimage.feature.canny(bkg_corr_blur, sigma=1)

bkg_corr_blur_thx = bkg_corr_blur>20

bkg_corr_skeleton = skimage.morphology.skeletonize(bkg_corr_blur_thx)

'''





