import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def displayImageAndAllHist(image, title = 'Bild'):
    """Display a grayscale image with its histogram and cumulative histogram.

    Parameters:
    - image (numpy.ndarray): uint8 image, grayscale
    - title (str): Plot title.
    """

    histo, edges = np.histogram(image, bins=256, range=(0, 256))
    cumHisto = np.cumsum(histo)
    
    # Matplotlib Plot mit Bild und Histogramm
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Bild anzeigen
    axs[0].imshow(image, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title(title)
    axs[0].axis('off')

    # Histogramm
    axs[1].bar(range(len(histo)), histo)
    axs[1].set_title('Histogramm')
    axs[1].set_xlabel('Grauwert')
    axs[1].set_ylabel('Anzahl')

    # Kumuliertes Histogramm
    axs[2].plot(range(len(cumHisto)), cumHisto, color='red')
    axs[2].set_ylim(0, image.size)
    axs[2].set_title('Kumuliertes Histogramm')
    axs[2].set_xlabel('Grauwert')
    axs[2].set_ylabel('Kumuliert')

    plt.tight_layout()
    plt.show()


def displayImagePlotly(image, title = 'Bild', height=500, width=500):
    fig = px.imshow(image, color_continuous_scale='gray', range_color=[0, 255], title = title)
    fig.update_layout(
    width = width,
    height = height)
    fig.show()