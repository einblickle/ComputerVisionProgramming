import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.plotting import displayImageAndAllHist 

def automatic_contrast_adjustment(image):
    """Führt eine automatische Kontrastanpassung durch, indem der Wertebereich des Bildes auf [0, 255] gestreckt wird."""
    a_low = np.min(image)
    a_high = np.max(image)
    a_min = 0
    a_max = 255
    
    # Kontrastanpassung durchführen
    adjusted_image = a_min + (image - a_low) * (a_max - a_min) / (a_high - a_low) 
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
    
    return adjusted_image



image_landscape = cv2.imread(r'images/landscape.png', cv2.IMREAD_GRAYSCALE)

displayImageAndAllHist(image_landscape, 'Originalbild:landscape.png')