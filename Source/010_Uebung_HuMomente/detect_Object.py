import cv2 
import numpy as np
from skimage.morphology import skeletonize
from skimage import data
from skimage.util import invert
import skimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



img = 255 - cv2.imread(r'C:\workspace\ComputerVision\Source\017_Kapitel_RegionenInBinaerbildern\Figs\motivation.png', cv2.IMREAD_ANYDEPTH)
labels = skimage.measure.label(img).astype('uint8')

obj = cv2.imread(r'.\images\spachtel_binary.png', cv2.IMREAD_ANYDEPTH)


numberOfObjects = np.max(labels)

huMomentsList = []

for region in np.arange(1, numberOfObjects+1):
    # isolate image
    img_tmp = np.zeros(img.shape, dtype='uint8')
    
    img_tmp[labels == region] = 1
    
    # calculate hu Moment
    moment_central = skimage.measure.moments_central(img_tmp)
    moment_normal = skimage.measure.moments_normalized(moment_central)
    moment_Hu = skimage.measure.moments_hu(moment_normal)
    moment_Hu_log =- np.sign(moment_Hu)*np.log(np.abs(moment_Hu))
    huMomentsList.append(list(moment_Hu_log))
    pass

huMoments = np.array(huMomentsList)

moment_central = skimage.measure.moments_central(obj / 255)
moment_normal = skimage.measure.moments_normalized(moment_central)
moment_Hu = skimage.measure.moments_hu(moment_normal)
moment_Hu_log =- np.sign(moment_Hu)*np.log(np.abs(moment_Hu))
objMoments = moment_Hu_log[np.newaxis,:]


diff = huMoments - objMoments
distance = np.linalg.norm(diff, axis =1)