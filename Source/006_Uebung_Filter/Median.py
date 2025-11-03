# package import
import numpy as np
import matplotlib.pyplot as plt
import cv2
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules import plotting

def median_filter(image, filter_size):

    # Bestimmen der Filterhalbgröße
    half_filter_size = filter_size // 2
    
    imageSizeX = image.shape[0]
    imageSizeY = image.shape[1]
    # Anwendung des Medianfilters
    image_filtered = np.zeros((imageSizeX- filter_size +1, imageSizeY - filter_size +1))
   
    for i in range(image_filtered.shape[0]):
        for j in range(image_filtered.shape[1]):
             image_ROI = image[i:i + filter_size, j:j + filter_size]
             valueList = image_ROI.flatten()
             image_filtered[i, j] = np.median(valueList)

    return image_filtered

# Bilddatei einladen
image = cv2.imread(r'images/clown_noise.png', cv2.IMREAD_GRAYSCALE)

#Datentyp:
imageDataType = image.dtype  ## ENTER YOUR CODE HERE
#Bildgröße
imageShape =  image.shape  ## ENTER YOUR CODE HERE

# Ausgabe der Daten
print(f'Datentyp des Bildes: {imageDataType}')
print(f'Größe des Bildes: {imageShape}')

# Anzeige des Originalbildes
plotting.displayImagePlotly(image, title='Originalbild mit Rauschen')




image_filtered = median_filter(image,5)