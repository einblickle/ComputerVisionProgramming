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

'''
Regionen:
Beiss-Zange == 9
Hammer == 3
Mutter == 4
Nagel == 5
'''


# Isolate Objects

img_BZange = img.copy()
img_BZange[labels != 9] = 0
img_BZange = img_BZange/255

img_Hammer = img.copy()
img_Hammer[labels != 3] = 0
img_Hammer = img_Hammer/255


img_Mutter = img.copy()
img_Mutter[labels != 4] = 0
img_Mutter = img_Mutter/255

img_Nagel = img.copy()
img_Nagel[labels != 5] = 0
img_Nagel = img_Nagel/255



# Calculate Moments
mom_Bzange_central = skimage.measure.moments_central(img_BZange)
mom_Bzange_normal = skimage.measure.moments_normalized(mom_Bzange_central)
mom_Bzange_Hu = skimage.measure.moments_hu(mom_Bzange_normal)
log_Bzange = - np.log(mom_Bzange_Hu)

mom_Hammer_central = skimage.measure.moments_central(img_Hammer)
mom_Hammer_normal = skimage.measure.moments_normalized(mom_Hammer_central)
mom_Hammer_Hu = skimage.measure.moments_hu(mom_Hammer_normal)
log_Hammer =- np.log(mom_Hammer_Hu)

mom_Mutter_central = skimage.measure.moments_central(img_Mutter)
mom_Mutter_normal = skimage.measure.moments_normalized(mom_Mutter_central)
mom_Mutter_Hu = skimage.measure.moments_hu(mom_Mutter_normal)
log_Mutter = - np.log(mom_Mutter_Hu)


mom_Nagel_central = skimage.measure.moments_central(img_Nagel)
mom_Nagel_normal = skimage.measure.moments_normalized(mom_Nagel_central)
mom_Nagel_Hu = skimage.measure.moments_hu(mom_Nagel_normal)
log_Nagel =- np.log(mom_Nagel_Hu)



# 2. Create a new figure and add a 3D subplot
fig = plt.figure()
# The 'projection="3d"' argument is crucial
ax = fig.add_subplot(111, projection='3d')

# 3. Use ax.scatter() to plot the point
# c='r' sets the color to red
# s=100 sets the size of the marker
col1 = 2
col2 = 4
col3 = 6

ax.scatter(log_Nagel[col1], log_Nagel[col2], log_Nagel[col3], c='r', marker='o', s=100, label = 'Nagel')
ax.scatter(log_Mutter[col1], log_Mutter[col2], log_Mutter[col3], c='b', marker='x', s=100, label = 'Mutter')
ax.scatter(log_Bzange[col1], log_Bzange[col2], log_Bzange[col3], c='black', marker='d', s=100, label = 'Zange')
ax.scatter(log_Hammer[col1], log_Hammer[col2], log_Hammer[col3], c='green', marker='v', s=100, label = 'Hammer')
#ax.scatter((log_Mutter[col1]-1), (log_Mutter[col2]-2), log_Mutter[col3], c='grey', marker='<', s=100, label = 'Unknown')


# 4. Set labels and title
ax.set_xlabel('Phi_3')
ax.set_ylabel('Phi_5')
ax.set_zlabel('Phi_7')
ax.set_title('Hu-Momente Phi_0 ... Phi_2')
ax.legend()
# 5. Set the limits for better visualization
#ax.set_xlim([0, 10])
#ax.set_ylim([0, 10])
#ax.set_zlim([0, 10])




plt.show()





'''

image = np.zeros((20, 20), dtype=np.uint8)

image[13:17, 13:17] = 1

image[10:12, 10:12] = 1

mu = skimage.measure.moments_central(image)

nu = skimage.measure.moments_normalized(mu)

muHu = skimage.measure.moments_hu(nu)

'''