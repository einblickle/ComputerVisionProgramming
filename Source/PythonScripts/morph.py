import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import data
from skimage.util import invert
import skimage


"""
#img = cv2.imread(r'.\img\rhino_detail.tif', cv2.IMREAD_ANYDEPTH)
#image_raw = image_raw[::-1,:]

img_er5x5 = skimage.morphology.binary_erosion(img, np.ones((5,5)))

img_er3x3 = skimage.morphology.binary_erosion(img, np.ones((3,3)))


img_dil5x5 = skimage.morphology.binary_dilation(img, np.ones((5,5)))

img_dil3x3 = skimage.morphology.binary_dilation(img, np.ones((3,3)))


img_er_disk1 = skimage.morphology.binary_erosion(img,skimage.morphology.disk(1)).astype('uint8')*255
img_er_disk2 = skimage.morphology.binary_erosion(img,skimage.morphology.disk(2)).astype('uint8')*255
img_er_disk2_5 = skimage.morphology.binary_erosion(img,skimage.morphology.disk(2.5)).astype('uint8')*255
img_er_disk3 = skimage.morphology.binary_erosion(img,skimage.morphology.disk(3)).astype('uint8')*255
img_er_disk4 = skimage.morphology.binary_erosion(img,skimage.morphology.disk(4)).astype('uint8')*255

cv2.imwrite(r'.\img\rhino_eroded_disk1.png', img_er_disk1)
cv2.imwrite(r'.\img\rhino_eroded_disk2.png', img_er_disk2)
cv2.imwrite(r'.\img\rhino_eroded_disk2_5.png', img_er_disk2_5)
cv2.imwrite(r'.\img\rhino_eroded_disk3.png', img_er_disk3)
cv2.imwrite(r'.\img\rhino_eroded_disk4.png', img_er_disk4)



img_dil_disk1 = skimage.morphology.binary_dilation(img,skimage.morphology.disk(1)).astype('uint8')*255
img_dil_disk2 = skimage.morphology.binary_dilation(img,skimage.morphology.disk(2)).astype('uint8')*255
img_dil_disk2_5 = skimage.morphology.binary_dilation(img,skimage.morphology.disk(2.5)).astype('uint8')*255
img_dil_disk3 = skimage.morphology.binary_dilation(img,skimage.morphology.disk(3)).astype('uint8')*255
img_dil_disk4 = skimage.morphology.binary_dilation(img,skimage.morphology.disk(4)).astype('uint8')*255

cv2.imwrite(r'.\img\rhino_dilated_disk1.png', img_dil_disk1)
cv2.imwrite(r'.\img\rhino_dilated_disk2.png', img_dil_disk2)
cv2.imwrite(r'.\img\rhino_dilated_disk2_5.png', img_dil_disk2_5)
cv2.imwrite(r'.\img\rhino_dilated_disk3.png', img_dil_disk3)
cv2.imwrite(r'.\img\rhino_dilated_disk4.png', img_dil_disk4)



img_dil_repeat1 = skimage.morphology.binary_dilation(img,skimage.morphology.disk(1)).astype('uint8')*255
img_dil_repeat2 = skimage.morphology.binary_dilation(img_dil_repeat1,skimage.morphology.disk(1)).astype('uint8')*255
img_dil_repeat3 = skimage.morphology.binary_dilation(img_dil_repeat2,skimage.morphology.disk(1)).astype('uint8')*255
img_dil_repeat4 = skimage.morphology.binary_dilation(img_dil_repeat3,skimage.morphology.disk(1)).astype('uint8')*255
img_dil_repeat5 = skimage.morphology.binary_dilation(img_dil_repeat4,skimage.morphology.disk(1)).astype('uint8')*255

cv2.imwrite(r'.\img\rhino_dilated_repeat1.png', img_dil_repeat1)
cv2.imwrite(r'.\img\rhino_dilated_repeat2.png', img_dil_repeat2)
cv2.imwrite(r'.\img\rhino_dilated_repeat3.png', img_dil_repeat3)
cv2.imwrite(r'.\img\rhino_dilated_repeat4.png', img_dil_repeat4)
cv2.imwrite(r'.\img\rhino_dilated_repeat5.png', img_dil_repeat5)
"""

img = 255 - cv2.imread(r'.\img\cat.png', cv2.IMREAD_ANYDEPTH)

img_dil = skimage.morphology.binary_dilation(img,skimage.morphology.disk(1)).astype('uint8')*255

img_er = skimage.morphology.binary_erosion(img,skimage.morphology.disk(1)).astype('uint8')*255

img_grad = (img_er - img_dil)*255


cv2.imwrite(r'.\img\cat_inv.png', img)
cv2.imwrite(r'.\img\cat_dil.png', img_dil)
cv2.imwrite(r'.\img\cat_er.png', img_er)
cv2.imwrite(r'.\img\cat_grad.png', img_grad)


img = 255 - cv2.imread(r'.\img\cat.png', cv2.IMREAD_ANYDEPTH)
