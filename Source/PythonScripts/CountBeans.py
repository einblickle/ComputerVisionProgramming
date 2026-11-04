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
from scipy.ndimage import uniform_filter


def repeatErosion(img, repetitions):
    eroded = img
    for iter in range(repetitions):
        eroded = skimage.morphology.isotropic_erosion(eroded, radius = 5)

    return eroded

def getAverageIntensityValue(image, mask):
    regionAV = np.zeros(np.max(mask)+1)
    for region in range(np.max(mask)+1):
        mean = np.mean(image[mask == region])
        regionAV[region] = mean
    return regionAV

def calculateAllCentroids(imageLabeled):

    # iterate over all regions
    noOfRegions = np.max(imageLabeled)+1
    centerOfRegions = np.zeros((noOfRegions, 2))
    for regionId in range(noOfRegions):
        centroid = skimage.measure.centroid(imageLabeled == regionId)
        centerOfRegions[regionId,:] = centroid
    return centerOfRegions




#img = skimage.io.imread(r'img/beans/beansMixed2.jpg', False)
img = skimage.io.imread(r'img/beans/many6.jpg', False)

fig = px.imshow(img)
fig.show()



# preprocess image Gaussian blur
img = uniform_filter(img, size=(25, 25, 1))


fig = px.imshow(img)
fig.show()

#change color to hsv

#img_hsv = skimage.color.rgb2hsv(img)
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


fig = px.imshow(img_hsv)
fig.update_traces(
    hovertemplate="<br>".join([
        "X: %{x}",
        "Y: %{y}",
        "<b>Color Values:</b>",
        "Hue: %{customdata[0]}",
        "Sat: %{customdata[1]}",
        "Value: %{customdata[2]}<extra></extra>"
    ]),
    customdata=img_hsv
)
fig.show()







# cut out color range of interest
img_hue = img_hsv[:,:,0]
img_sat = img_hsv[:,:,1]
img_val = img_hsv[:,:,2]

# filter hue sat Value

hue_low = 70
hue_high = 120

sat_low = 127
sat_high = 255

val_low = 125
val_high = 255


mask_hue = 1-((img_hsv[:,:,0]>=hue_low)&(hue_high>=img_hsv[:,:,0]))

mask_sat = 1-((img_hsv[:,:,1]>=sat_low)&(sat_high>=img_hsv[:,:,1]))



mask = mask_hue
mask=mask.astype('uint8')

fig = px.imshow(mask)
fig.show()


kernel = skimage.morphology.disk(5)

#mask = cv2.erode(mask, kernel, iterations=5)


distance = scipy.ndimage.distance_transform_edt(mask > 0)

# 2. Lokale Maxima (Kerne der getrennten Objekte) finden
coords = skimage.feature.peak_local_max(distance, footprint=np.ones((3, 3)),threshold_abs
                                                                            =70, min_distance=170, labels = mask )
markers_mask = np.zeros(distance.shape, dtype=bool)
markers_mask[tuple(coords.T)] = True
markers, _ = scipy.ndimage.label(markers_mask)

# 3. Watershed anwenden, um die Brücke zu schneiden
labels = skimage.segmentation.watershed(-distance, markers, mask=(mask > 0))

# jetzt hat man die Bohnen erkannt

centerOfRegions = calculateAllCentroids(labels)

# Jetzt will man die Bohnensorte abtrennen.
avIntInRegion = getAverageIntensityValue(img_hsv[:,:,2],labels)

isBeanRed = avIntInRegion<140



posBeanRed = centerOfRegions[isBeanRed==True,:]
posBeansNotRed = centerOfRegions[isBeanRed==False,:]

# Show original image with overlay
fig = px.imshow(img_hue)
# 3. Das Overlay als Scatter-Trace hinzufügen
fig.add_trace(
    go.Scatter(
        x=posBeanRed[:,1],
        y=posBeanRed[:,0],
        mode='markers+text',
        marker=dict(size=14, color='red', symbol='cross'),
        textposition='top center',
        name='Schwerpunkte'
    )
)


fig.add_trace(
    go.Scatter(
        x=posBeansNotRed[:,1],
        y=posBeansNotRed[:,0],
        mode='markers+text',
        marker=dict(size=14, color='blue', symbol='cross'),
        textposition='top center',
        name='Schwerpunkte'
    )
)


fig.show()







'''
kernel = skimage.morphology.disk(5)

eroded = cv2.erode(mask, kernel, iterations=15)
# Rekonstruktion des Originalbildes basierend auf dem erodierten Seed
erodedb = eroded >0
maskb = mask>0
reconstructed = skimage.morphology.reconstruction(erodedb, maskb, method='dilation')
'''
#####
# test watershed
####
'''
distance = scipy.ndimage.distance_transform_edt(mask)
coords = skimage.feature.peak_local_max(distance, footprint=np.ones((3, 3)), labels=mask)
mask1 = np.zeros(distance.shape, dtype=bool)
mask1[tuple(coords.T)] = True
markers, _ = scipy.ndimage.label(mask1)
labels = skimage.segmentation.watershed(-distance, markers, mask=mask)
'''