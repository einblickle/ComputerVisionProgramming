import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import skimage


from modules.plotting import displayImageAndAllHist 

def autoContrast(image):
    """Führt eine automatische Kontrastanpassung durch, indem der Wertebereich des Bildes auf [0, 255] gestreckt wird."""
    a_low = np.min(image)
    a_high = np.max(image)
    a_min = 0
    a_max = 255
    
    # Kontrastanpassung durchführen
    adjusted_image = a_min + (image - a_low) * ((a_max - a_min) / (a_high - a_low)) 
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
    
    return adjusted_image


def autoContrastSat(image, satPixPercentage):
    """Führt eine automatische Kontrastanpassung durch, indem der Wertebereich des Bildes auf [0, 255] gestreckt wird."""
    a_low = np.percentile(image, satPixPercentage)
    a_high = np.percentile(image, (100-satPixPercentage))
    a_min = 0
    a_max = 255
    
    # Kontrastanpassung durchführen
    adjusted_image = a_min + (image - a_low) * ((a_max - a_min) / (a_high - a_low)) 
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
    
    return adjusted_image


plt.close('all')


image_landscape = cv2.imread(r'images/landscape.png', cv2.IMREAD_GRAYSCALE)

displayImageAndAllHist(image_landscape, 'Originalbild:landscape.png')

image_landscape_autocontrast = autoContrast(image_landscape)

displayImageAndAllHist(image_landscape_autocontrast, 'Autocontrast')

image_landscape_autocontrastSat = autoContrastSat(image_landscape, 0.5)

displayImageAndAllHist(image_landscape_autocontrastSat, 'Autocontrast')


image_whatIsThat = cv2.imread(r'images/unknown.png', cv2.IMREAD_GRAYSCALE)
image_whatIsThat_autocontrast = autoContrast(image_whatIsThat)
image_whatIsThat_autocontrastSat = autoContrastSat(image_whatIsThat,1)
displayImageAndAllHist(image_whatIsThat_autocontrastSat, 'Autocontrast')


imageEqualized = skimage.exposure.equalize_adapthist(image_landscape, clip_limit=0.3)
displayImageAndAllHist(imageEqualized, 'imageEqualized')