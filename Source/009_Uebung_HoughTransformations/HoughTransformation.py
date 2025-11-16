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

def calculateCircAccumulator(image, radius):
    
    template = createCircleImage(radius)
    halfWidth = template.shape[0]//2
    accumulator = np.zeros(image.shape).astype('int')

    edges = skimage.feature.canny(img, sigma=2)

    indices = np.where(edges ==1)
    for row, col in zip(*indices):
        accumulator[row-halfWidth:row+halfWidth+1,col-halfWidth:col+halfWidth+1] += template
        
    return accumulator





img = cv2.imread(r'images/circles.png', cv2.IMREAD_GRAYSCALE)
circRadius = 50

accumulator = calculateCircAccumulator(img, circRadius)

maxIndex_Y, maxIndex_X = np.where(accumulator >= 0.9*np.max(accumulator))




fig = px.imshow(
    img, 
    title="Plotly Express Image Display (px.imshow)",
    # Optional: Adjust the aspect ratio of the plot
    aspect="equal",
    color_continuous_scale='gray'
)
for posX, posY in zip(maxIndex_Y, maxIndex_X):
    fig.add_shape(type="circle",
        x0=posX-circRadius,
        y0=posY-circRadius,
        x1=posX+circRadius,
        y1=posY+circRadius,
        line=dict(
            color="Red",
            width=4,
        ),
        
    )
fig.update_shapes(dict(xref='x', yref='y'))
fig.show()