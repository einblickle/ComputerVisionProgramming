import numpy as np
import cv2
import plotly.express as px
import plotly.graph_objects as go
import skimage



center = (1659, 1506)
radius = 1280

image = cv2.imread(r'registration_full.tiff', cv2.IMREAD_ANYDEPTH)
mask = np.zeros(image.shape)
bkg = np.ones(image.shape).astype('uint16')*12000
mask = cv2.circle(mask, center = center, radius = radius, color = (1,1,1), thickness = -1)
mask = cv2.blur(mask, (51, 51))
mask = cv2.blur(mask, (51, 51))
mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX).astype('float')
maskedImage = image * mask + (1-mask)*bkg

maskedImage = maskedImage[200:2841, 330:2971]

# cover marks

maskedImage[1286:(1286+30), 1341:(1341+30)]=maskedImage[1200:(1200+30), 1499:(1499+30)]

maskedImage[1099:(1099+30), 1343:(1343+30)]=maskedImage[1200:(1200+30), 1499:(1499+30)]

maskedImage[1285:(1285+30), 1931:(1931+30)]=maskedImage[1200:(1200+30), 1499:(1499+30)]







## ROI

maskedImage_16Bit = cv2.normalize(maskedImage, None, 0, (2**16)-1, cv2.NORM_MINMAX).astype('uint16')
cv2.imwrite(r'registrationImage_16Bit.png', maskedImage_16Bit)

maskedImage = cv2.normalize(maskedImage, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

cv2.imwrite(r'registrationImage_8Bit.png', maskedImage)


# export image
image = cv2.imread(r'ConcCircTest.tiff', cv2.IMREAD_ANYDEPTH)
image_16Bit = cv2.normalize(image, None, 0, (2**16)-1, cv2.NORM_MINMAX).astype('uint16')
cv2.imwrite(r'concCirc_16Bit.png', image_16Bit)

image_8Bit = cv2.normalize(image, None, 0, (2**8)-1, cv2.NORM_MINMAX).astype('uint8')
cv2.imwrite(r'concCirc_8Bit.png', image_8Bit)