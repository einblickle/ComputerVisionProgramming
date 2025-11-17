import numpy as np
import cv2
import plotly.express as px
import plotly.graph_objects as go
import skimage



center = (1686, 1500)
radius = 1300

image = cv2.imread(r'registration_full.tiff', cv2.IMREAD_ANYDEPTH)





mask = np.ones(image.shape) * 20
mask = cv2.circle(mask, center = center, radius = radius, color = (255,255,255), thickness = -1)
mask = cv2.blur(mask, (51, 51))
mask = cv2.blur(mask, (51, 51))


mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX).astype('float')

maskedImage = image * mask


maskedImage = cv2.normalize(maskedImage, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
maskedImage = cv2.medianBlur(maskedImage,3)


# Global Threshold
disk = skimage.morphology.disk(71)

min_image = skimage.filters.rank.minimum(maskedImage, disk)
max_image = skimage.filters.rank.maximum(maskedImage, disk)

normalImage = (maskedImage-min_image) / (max_image+1)*255
thx = normalImage>50




# canny Filering
canny_filtered = skimage.feature.canny(maskedImage, sigma=2)
dilated = skimage.morphology.binary_dilation(canny_filtered, np.ones((5,5)))

eroded = skimage.morphology.binary_erosion(dilated, np.ones((5,5)))

skeleton = skimage.morphology.skeletonize(eroded)

radii = np.array([8, 25, 43])
radii = np.array([8])
H = skimage.transform.hough_circle(skeleton, radii)

accum, cx, cy, rad = skimage.transform.hough_circle_peaks(H, [radii, ] )


# local adaptive threshold
block_size = 151
local_thresh = skimage.filters.threshold_local(normalImage, block_size, offset=-20)
binary_local = normalImage > local_thresh
skeleton2 = skimage.morphology.skeletonize(binary_local)


'''

min_image_b = cv2.blur(min_image, (21,21))
#bkg_corr_blur = cv2.GaussianBlur(bkg_corr,(9,9), 3)

bkg_corr_blur = cv2.medianBlur(bkg_corr, 3)

laplace_filtered = skimage.filters.laplace(bkg_corr_blur, ksize=7)

canny_filtered = skimage.feature.canny(bkg_corr_blur, sigma=1)

bkg_corr_blur_thx = bkg_corr_blur>20

bkg_corr_skeleton = skimage.morphology.skeletonize(bkg_corr_blur_thx)



img_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

radius = 25
footprint = skimage.morphology.disk(radius)

local_otsu = skimage.filters.rank.otsu(img_8bit, footprint)
thx = img_8bit>=local_otsu


block_size = 151
local_thresh = skimage.filters.threshold_local(img_8bit, block_size, offset=-20)
binary_local = img_8bit > local_thresh
'''