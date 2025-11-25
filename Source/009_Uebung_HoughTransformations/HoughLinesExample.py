import numpy as np
import skimage
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm

img = cv2.imread(r'images/road2.jpg', cv2.IMREAD_GRAYSCALE)

imgBlur = cv2.medianBlur(img, 11)

sobX = cv2.Sobel(imgBlur,cv2.CV_16SC1, 1, 0, 3)
sobY = cv2.Sobel(imgBlur,cv2.CV_16SC1, 0, 1, 3)


edges = cv2.Canny(imgBlur, 200,70, 7)

h, theta, d = skimage.transform.hough_line(edges)
accum, angles, dist = skimage.transform.hough_line_peaks(h, theta, d, min_distance=200, threshold =0.4*np.max(h))

# drawings
img_col = skimage.color.gray2rgb(img)

fig, (ax) = plt.subplots(1,1)

ax.imshow(img, cmap=cm.gray)
ax.set_title('Detected Lines')
ax.set_axis_off()

for _, angle, dist in zip(accum, angles, dist):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    ax.axline((x0, y0), slope=np.tan(angle + np.pi / 2), color='red')

plt.show()