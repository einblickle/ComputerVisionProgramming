import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

image_baboon=cv2.imread('./Images/baboon.png', cv2.IMREAD_COLOR)
# image taklen from https://github.com/scijs/baboon-image/blob/master/baboon.png
# image is loaded as BRG image. convert to RGB
image_baboon = cv2.cvtColor(image_baboon, cv2.COLOR_BGR2RGB)



nose = cv2.inRange(image_baboon, (150, 0, 0), (255, 80, 80))



image_baboon_nose = image_baboon.copy()
image_baboon_nose[nose == 0] = 0



# konvertiere RGB zu HSV
image_baboon_hsv = cv2.cvtColor(image_baboon, cv2.COLOR_RGB2HSV)

## Verändern Sie die Filterwerte durch Anpassen der unteren Schranke und der oberen Schranke
jaw = cv2.inRange(image_baboon_hsv, (100, 0, 150), (105, 150, 255))

