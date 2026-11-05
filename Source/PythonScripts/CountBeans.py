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



def getElipticity(imageLabeled):
    
    noOfRegions = np.max(imageLabeled)
    elipticityOfRegion = np.zeros(noOfRegions+1)
    regions = skimage.measure.regionprops(imageLabeled)
    for regionId in range(noOfRegions):
        reg = regions[regionId]
        major_axis = reg.major_axis_length
        minor_axis = reg.minor_axis_length
        ellipticity = 1.0 - (minor_axis / major_axis) if major_axis > 0 else 0.0
        elipticityOfRegion[regionId+1] = ellipticity
    return elipticityOfRegion


#img = skimage.io.imread(r'img/beans/AllBeans1.jpg', False)
img = skimage.io.imread(r'img/beans/many1.jpg', False)

fig = px.imshow(img)
fig.show()



# preprocess image Gaussian blur
img_in = img
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
hue_high = 105

sat_low = 200
sat_high = 255

val_low = 0
val_high = 0


mask_hue = ((img_hsv[:,:,0]>=hue_low)&(hue_high>=img_hsv[:,:,0]))

mask_sat = ((img_hsv[:,:,1]>=sat_low)&(sat_high>=img_hsv[:,:,1]))

mask_val = ((img_hsv[:,:,2]>=val_low)&(val_high>=img_hsv[:,:,2]))



mask = 1-(mask_hue | mask_sat | mask_val)
mask=mask.astype('uint8')

footprint = skimage.morphology.disk(5)
mask = skimage.morphology.opening(mask, footprint)
mask = skimage.morphology.closing(mask, footprint)


fig = px.imshow(mask)
fig.show()


kernel = skimage.morphology.disk(5)

#mask = cv2.erode(mask, kernel, iterations=5)


distance = scipy.ndimage.distance_transform_edt(mask > 0)

# 2. Lokale Maxima (Kerne der getrennten Objekte) finden
coords = skimage.feature.peak_local_max(distance, footprint=np.ones((3, 3)),threshold_abs
                                                                            =70, min_distance=200, labels = mask )
markers_mask = np.zeros(distance.shape, dtype=bool)
markers_mask[tuple(coords.T)] = True
markers, _ = scipy.ndimage.label(markers_mask)

# 3. Watershed anwenden, um die Brücke zu schneiden
labels = skimage.segmentation.watershed(-distance, markers, mask=(mask > 0))

# jetzt hat man die Bohnen erkannt

centerOfRegions = calculateAllCentroids(labels)

# Jetzt will man die Bohnensorte abtrennen.
avIntInRegion = getAverageIntensityValue(img_hsv[:,:,2],labels)


# Ellitizität der Bereiche bestimmen
ellipticityInRegion = getElipticity(labels)

isRegionDark = avIntInRegion<140
isRegionBright = avIntInRegion >140
isRegionRound = ellipticityInRegion < 0.35

isRedBean = isRegionDark
isWhiteBean = isRegionBright & (~isRegionRound)
isChickpea = isRegionBright & isRegionRound

posBeanRed = centerOfRegions[isRedBean,:]
posBeanWhite = centerOfRegions[isWhiteBean,:]
posChickpea = centerOfRegions[isChickpea,:]

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
        x=posBeanWhite[:,1],
        y=posBeanWhite[:,0],
        mode='markers+text',
        marker=dict(size=14, color='blue', symbol='cross'),
        textposition='top center',
        name='Schwerpunkte'
    )
)


fig.show()


# Show original image with overlay
fig = px.imshow(img_in)
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
        x=posBeanWhite[:,1],
        y=posBeanWhite[:,0],
        mode='markers+text',
        marker=dict(size=14, color='blue', symbol='star'),
        textposition='top center',
        name='Schwerpunkte'
    )
)

fig.add_trace(
    go.Scatter(
        x=posChickpea[:,1],
        y=posChickpea[:,0],
        mode='markers+text',
        marker=dict(size=14, color='green', symbol='cross'),
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