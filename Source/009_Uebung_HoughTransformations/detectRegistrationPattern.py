import numpy as np
import cv2
import plotly.express as px
import plotly.graph_objects as go
import skimage
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def preprocessImageThx(image_raw):
    denoised_img = skimage.restoration.denoise_tv_chambolle(image_raw, weight=0.1)
    denoised_img_cp = denoised_img.copy()
    
    
    denoised_img = cv2.normalize(denoised_img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    
    
    disk = skimage.morphology.disk(71)
    
    min_image = skimage.filters.rank.minimum(denoised_img, disk)
    max_image = skimage.filters.rank.maximum(denoised_img, disk)
    
    normalImage = (denoised_img-min_image) / (max_image)*255
    
    normalImage = denoised_img
    block_size = 151
    local_thresh = skimage.filters.threshold_local(normalImage, block_size, offset=-10)
    thx = normalImage > local_thresh
    
    
    skeleton = skimage.morphology.skeletonize(thx)
    return skeleton

def preprocessImageMorph(image_raw):
    canny_filtered = skimage.feature.canny(image_raw, sigma=2)
    dilated = skimage.morphology.binary_dilation(canny_filtered, np.ones((5,5)))
    
    #eroded = skimage.morphology.binary_erosion(dilated, np.ones((5,5)))
    
    skeleton = skimage.morphology.skeletonize(dilated)

    return skeleton




#####################
#####################

image_raw = cv2.imread(r'.\images\registrationImage_8Bit.png', cv2.IMREAD_ANYDEPTH)

if True:
    #skeleton1 = preprocessImageThx(image_raw)
    skeleton = preprocessImageMorph(image_raw)

radius1 = 43
hough_radii = np.array([radius1])
hough_res = skimage.transform.hough_circle(skeleton, hough_radii)
accums1, cx1, cy1, radii1 = skimage.transform.hough_circle_peaks(hough_res,
                                                             hough_radii,
                                                             min_xdistance = 70,
                                                             min_ydistance = 70,
                                                             threshold = 0,
                                                             num_peaks=255)

radius2 = 26
hough_radii = np.array([radius2])
hough_res = skimage.transform.hough_circle(skeleton, hough_radii)
accums2, cx2, cy2, radii2 = skimage.transform.hough_circle_peaks(hough_res,
                                                             hough_radii,
                                                             min_xdistance = 70,
                                                             min_ydistance = 70,
                                                             threshold = 0,
                                                             num_peaks=255)


radius3 = 8
hough_radii = np.array([radius3])
hough_res = skimage.transform.hough_circle(skeleton, hough_radii)
accums3, cx3, cy3, radii3 = skimage.transform.hough_circle_peaks(hough_res,
                                                             hough_radii,
                                                             min_xdistance = 70,
                                                             min_ydistance = 70,
                                                             threshold = 0,
                                                             num_peaks=255)


print('finished hough')

'''
fig_detected = px.imshow(
    image_raw, 
    title="Plotly Express Image Display (px.imshow)",
    # Optional: Adjust the aspect ratio of the plot
    aspect="equal",
    color_continuous_scale='gray'
)

for center_y, center_x, radius in zip(cy1, cx1, radii1):
    # add a the detected circle as shape to that figure
    fig_detected.add_shape(type="circle",
        x0=center_x-radius,
        y0=center_y-radius,
        x1=center_x+radius,
        y1=center_y+radius,
        line=dict(
            color="Chartreuse",
            width=1))

for center_y, center_x, radius in zip(cy2, cx2, radii2):
    # add a the detected circle as shape to that figure
    fig_detected.add_shape(type="circle",
        x0=center_x-radius,
        y0=center_y-radius,
        x1=center_x+radius,
        y1=center_y+radius,
        line=dict(
            color="Red",
            width=1))

for center_y, center_x, radius in zip(cy3, cx3, radii3):
    # add a the detected circle as shape to that figure
    fig_detected.add_shape(type="circle",
        x0=center_x-radius,
        y0=center_y-radius,
        x1=center_x+radius,
        y1=center_y+radius,
        line=dict(
            color="Blue",
            width=1))


fig_detected.show()

'''

fig, ax = plt.subplots(1,1)

# Display the image using imshow
ax.imshow(image_raw, cmap='gray', vmin=0, vmax=255)

for center_y, center_x, radius in zip(cy3, cx3, radii3):
    circle = Circle((center_x, center_y), radius,
                    color='cyan',          # Cyan color
                    linestyle='-',         # Solid line
                    linewidth=3,           # Thicker line
                    alpha=0.8,             # Slightly transparent
                    fill=False)            # Do not fill the circle
    
    # Add the circle to the axes
    ax.add_patch(circle)
    
for center_y, center_x, radius in zip(cy2, cx2, radii2):
    circle = Circle((center_x, center_y), radius,
                    color='red',          # Cyan color
                    linestyle='-',         # Solid line
                    linewidth=3,           # Thicker line
                    alpha=0.8,             # Slightly transparent
                    fill=False)            # Do not fill the circle
    
    # Add the circle to the axes
    ax.add_patch(circle)
    
for center_y, center_x, radius in zip(cy1, cx1, radii1):
    circle = Circle((center_x, center_y), radius,
                    color='green',          # Cyan color
                    linestyle='-',         # Solid line
                    linewidth=3,           # Thicker line
                    alpha=0.8,             # Slightly transparent
                    fill=False)            # Do not fill the circle
    
    # Add the circle to the axes
    ax.add_patch(circle)

