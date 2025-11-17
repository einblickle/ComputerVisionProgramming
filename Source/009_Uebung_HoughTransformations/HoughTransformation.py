import numpy as np
import skimage
import cv2
import matplotlib.pyplot as plt
import plotly.express as px



def createCircleImage(radius):
    center = (radius, radius)
    img = np.zeros((2*radius+1,2*radius+1))
    #img1 = cv2.circle(img, center, radius, color = 1, thickness = 1)
    rr, cc  = skimage.draw.circle_perimeter(radius, radius, radius)
    img[rr,cc] = 1

    return img.astype('int')
'''
def calculateCircAccumulator(image, radius):
    
    template = createCircleImage(radius)
    halfWidth = template.shape[0]//2
    accumulator = np.zeros(image.shape).astype('int')

    edges = skimage.feature.canny(img, sigma=2)

    indices = np.where(edges ==1)
    for row, col in zip(*indices):
        accumulator[row-halfWidth:row+halfWidth+1,col-halfWidth:col+halfWidth+1] += template
        
    return accumulator
'''

def calculateCircAccumulator(img_edges, img_circ):
    
    
    halfWidth = img_circ.shape[0]//2
    accumulator = np.zeros(img_edges.shape).astype('int')

    indices = np.where(img_edges ==1)
    for row, col in zip(*indices):
        accumulator[row-halfWidth:row+halfWidth+1,col-halfWidth:col+halfWidth+1] += img_circ
        
    return accumulator





img = cv2.imread(r'images/circles.png', cv2.IMREAD_GRAYSCALE)
img_edges = skimage.feature.canny(img, sigma=2)


circRadius = 50
img_circ = createCircleImage(circRadius)
accumulator = calculateCircAccumulator(img_edges, img_circ)
maxIndex50_Y, maxIndex50_X = np.where(accumulator >= 0.9*np.max(accumulator))

circRadius = 55
img_circ = createCircleImage(circRadius)
accumulator = calculateCircAccumulator(img_edges, img_circ)
maxIndex55_Y, maxIndex55_X = np.where(accumulator >= 0.9*np.max(accumulator))

circRadius = 45
img_circ = createCircleImage(circRadius)
accumulator = calculateCircAccumulator(img_edges, img_circ)
maxIndex45_Y, maxIndex45_X = np.where(accumulator >= 0.9*np.max(accumulator))


fig = px.imshow(
    img, 
    title="Plotly Express Image Display (px.imshow)",
    # Optional: Adjust the aspect ratio of the plot
    aspect="equal",
    color_continuous_scale='gray'
)

circRadius = 50
for posX, posY in zip(maxIndex50_Y, maxIndex50_X):
    fig.add_shape(type="circle",
        x0=posX-circRadius,
        y0=posY-circRadius,
        x1=posX+circRadius,
        y1=posY+circRadius,
        line=dict(
            color="Chartreuse",
            width=4))

circRadius = 55
for posX, posY in zip(maxIndex55_Y, maxIndex55_X):
    fig.add_shape(type="circle",
        x0=posX-circRadius,
        y0=posY-circRadius,
        x1=posX+circRadius,
        y1=posY+circRadius,
        line=dict(
            color="Red",
            width=4))

circRadius = 45
for posX, posY in zip(maxIndex45_Y, maxIndex45_X):
    fig.add_shape(type="circle",
        x0=posX-circRadius,
        y0=posY-circRadius,
        x1=posX+circRadius,
        y1=posY+circRadius,
        line=dict(
            color="Cyan",
            width=4))


        
fig.update_shapes(dict(xref='x', yref='y'))
fig.show()


## create circle coordinates and draw line directly into image

rr, cc  = skimage.draw.circle_perimeter(maxIndex45_Y[0], maxIndex45_X[0], 45)

img_rgb = skimage.color.gray2rgb(img)
img_rgb[rr,cc,:]=[127,255,0]

fig = px.imshow(
    img_rgb, 
    title="Plotly Express Image Display (px.imshow)",
    # Optional: Adjust the aspect ratio of the plot
    aspect="equal",
    color_continuous_scale='rgb'
)

fig.show()