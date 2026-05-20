# package import
import numpy as np
import matplotlib.pyplot as plt
import cv2
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules import plotting
# Load Modules
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage

testImg  = np.zeros((600,600)).astype('uint8')

#circles
cv2.circle(testImg, (100, 150), 50, 255, -1, cv2.FILLED)
cv2.circle(testImg, (250, 150), 6, 255, -1, cv2.FILLED)
cv2.circle(testImg, (400, 150), 20, 255, -1, cv2.FILLED)

#points
testImg[150, 200] = 255
testImg[150, 325] = 255

#lines
cv2.line(testImg, (100, 310), (100, 390), 255, 4, cv2.LINE_8)
cv2.line(testImg, (60, 350), (140, 350), 255, 4, cv2.LINE_8)

cv2.line(testImg, (200, 410), (200, 490), 255, 1, cv2.LINE_8)
cv2.line(testImg, (160, 450), (240, 450), 255, 1, cv2.LINE_8)

cv2.line(testImg, (370, 320), (430, 370), 255, 1, cv2.LINE_8)
cv2.line(testImg, (370, 370), (430, 320), 255, 1, cv2.LINE_8)

cv2.line(testImg, (470, 320), (530, 370), 255, 2, cv2.LINE_8)
cv2.line(testImg, (470, 370), (530, 320), 255, 2, cv2.LINE_8)

cv2.line(testImg, (370, 470), (430, 520), 255, 3, cv2.LINE_8)
cv2.line(testImg, (370, 520), (430, 470), 255, 3, cv2.LINE_8)



