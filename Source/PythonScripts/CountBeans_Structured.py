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



def getAverageIntensityValue(image, mask):
    noOfRegions = np.max(mask)+1
    regionAV = np.zeros(noOfRegions-1)
    for region in range(1,noOfRegions):
        mean = np.mean(image[mask == region])
        regionAV[region-1] = mean
    return regionAV

def calculateAllCentroids(imageLabeled):

    # iterate over all regions
    noOfRegions = np.max(imageLabeled)+1
    centerOfRegions = np.zeros((noOfRegions-1, 2))
    for regionId in range(1, noOfRegions):
        centroid = skimage.measure.centroid(imageLabeled == regionId)
        centerOfRegions[regionId-1,:] = centroid
    return centerOfRegions



def getElipticity(imageLabeled):
    
    noOfRegions = np.max(imageLabeled)+1
    elipticityOfRegion = np.zeros(noOfRegions-1)
    regions = skimage.measure.regionprops(imageLabeled)
    for regionId in range(1, noOfRegions):
        reg = regions[regionId-1]
        major_axis = reg.major_axis_length
        minor_axis = reg.minor_axis_length
        ellipticity = 1.0 - (minor_axis / major_axis) if major_axis > 0 else 0.0
        elipticityOfRegion[regionId-1] = ellipticity
    return elipticityOfRegion



def buildFeatureVector(image, imageLabeled):
    ellipticityInRegion = getElipticity(imageLabeled)
    avIntInRegion = getAverageIntensityValue(image, imageLabeled)
    avIntInRegion = avIntInRegion/255
    featureVector = np.hstack((avIntInRegion[:,np.newaxis], ellipticityInRegion[:,np.newaxis]))
    return featureVector
    
    



def getSorte(featReference, featureVector):
    row, col = featureVector.shape
    whatKindOfBean = np.zeros((row))*np.nan
    for region in range(row):
        diff = featReference-featureVector[region,:]
        distance = np.linalg.norm(diff, axis=1)
        objectNumber = np.argmin(distance)
        whatKindOfBean[region] = objectNumber
    return whatKindOfBean


def imagePreprocessing(img):
    img = uniform_filter(img, size=(25, 25, 1))
    return img


def seperateGreenBkg(img_hsv, low, high):

    mask_hue = 1-((img_hsv[:,:,0]>low[0])&(img_hsv[:,:,0]<high[0]))
    
    mask_sat = 1-((img_hsv[:,:,1]>low[1])&(img_hsv[:,:,1]<high[1]))
    
    mask_val = ((img_hsv[:,:,2]>low[2])&(img_hsv[:,:,2]<high[2]))
    
    mask =(mask_hue & mask_sat & mask_val)
    mask=mask.astype('uint8')

    '''
    mask = cv2.inRange(img_hsv, low, high, )
    der obige code funktioniert nicht mit einem einzigen cv2.inRange befehl.
    '''
    return mask


def watershedSegmentation(mask):
    kernel = skimage.morphology.disk(5)
    
    #mask = cv2.erode(mask, kernel, iterations=5)
    
    distance = scipy.ndimage.distance_transform_edt(mask > 0)
    
    # 2. Lokale Maxima (Kerne der getrennten Objekte) finden
    coords = skimage.feature.peak_local_max(distance, footprint=np.ones((155, 155)),threshold_abs
                                                                                =50, min_distance=150, labels = mask )
    markers_mask = np.zeros(distance.shape, dtype=bool)
    markers_mask[tuple(coords.T)] = True
    markers, _ = scipy.ndimage.label(markers_mask)
    
    # 3. Watershed anwenden, um die Brücke zu schneiden
    segments = skimage.segmentation.watershed(-distance, markers, mask=(mask > 0))
    
    return segments

def imageSegmentation(mask):

    segments = watershedSegmentation(mask)
    #segments = skimage.measure.label(mask)
    #segments = cv2.connectedComponents(mask)[1]
    return segments
    

####
### Read Image
####


#img = skimage.io.imread(r'img/beans/KidneyBeans.jpg', False)
#img = skimage.io.imread(r'img/beans/Kichererbsen.jpg', False)
img = skimage.io.imread(r'img/beans/AllBeans1.jpg', False)
#img = skimage.io.imread(r'img/beans/many2.jpg', False)

img_in = img

## Show Image
fig = px.imshow(img)
fig.show()


####
# Image Preprocessing
####
img = imagePreprocessing(img)

## Show Image
fig = px.imshow(img)
fig.show()


####
# Main Image Processing
####
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

## Show Image
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


# seperate foreground and background
low = np.array([20,200,40])
high = np.array([105, 255, 255])
mask = seperateGreenBkg(img_hsv, low, high)


# clean regions
footprint = skimage.morphology.disk(25)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, footprint)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, footprint)

# image segmentation

img_Labeled = imageSegmentation(mask)

fig = px.imshow(img_Labeled, color_continuous_scale="Viridis")
fig.show()

# Zentren der jeweiligen Regionen berechnen
centerOfRegions = calculateAllCentroids(img_Labeled)

# Mittlere Helligkeit der jeweiligen Region
avIntInRegion = getAverageIntensityValue(img_hsv[:,:,2],img_Labeled)



# Ellitizität der Bereiche bestimmen
ellipticityInRegion = getElipticity(img_Labeled)



## build featureVector

featureVector = buildFeatureVector(img_hsv[:,:,2], img_Labeled)

## build Feature Reference

featureBean = np.array([0.3, 0.5])
featureWhiteBean = np.array([0.8, 0.5])
featureChickPea = np.array([0.8, 0.08])

featureReference=np.vstack((featureBean[np.newaxis,:],
                            featureWhiteBean[np.newaxis,:],
                            featureChickPea[np.newaxis,:]))


sorte = getSorte(featureReference, featureVector)



fig = px.imshow(img_in)
# 3. Das Overlay als Scatter-Trace hinzufügen

fig.add_trace(
    go.Scatter(
        x=centerOfRegions[sorte == 0,1],
        y=centerOfRegions[sorte == 0,0],
        mode='markers+text',
        marker=dict(size=14, color='red', symbol='cross'),
        textposition='middle center',
        name='Kidney-Bohne',
        showlegend = True
    )
)

fig.add_trace(
    go.Scatter(
        x=centerOfRegions[sorte == 1,1],
        y=centerOfRegions[sorte == 1,0],
        mode='markers+text',
        marker=dict(size=14, color='blue', symbol='star'),
        textposition='middle center',
        name='Weiße Bohne',
        showlegend = True
    )
)

fig.add_trace(
    go.Scatter(
        x=centerOfRegions[sorte == 2,1],
        y=centerOfRegions[sorte == 2,0],
        mode='markers+text',
        marker=dict(size=14, color='green', symbol='cross'),
        textposition='middle center',
        name='Kichererbse',
        showlegend = True
    )
)

fig.show()




######
######
'''
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
fig = px.imshow(img_in)
# 3. Das Overlay als Scatter-Trace hinzufügen
fig.add_trace(
    go.Scatter(
        x=posBeanRed[:,1],
        y=posBeanRed[:,0],
        mode='markers+text',
        marker=dict(size=14, color='red', symbol='cross'),
        textposition='middle center',
        name='Kidney-Bohne',
        showlegend = True
    )
)

fig.add_trace(
    go.Scatter(
        x=posBeanWhite[:,1],
        y=posBeanWhite[:,0],
        mode='markers+text',
        marker=dict(size=14, color='blue', symbol='star'),
        textposition='middle center',
        name='Weiße-Bohne',
        showlegend = True
    )
)

fig.add_trace(
    go.Scatter(
        x=posChickpea[:,1],
        y=posChickpea[:,0],
        mode='markers+text',
        marker=dict(size=14, color='green', symbol='cross'),
        textposition='middle center',
        name='Kichererbse',
        showlegend = True
    )
)

fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
    )
)


fig.show()
'''