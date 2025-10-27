import numpy as np
import cv2
import plotly.express as px
import plotly.graph_objects as go

def addTemplateToIMG(image, template, posX, posY):
    img = image.copy()
    sizeX= template.shape[1]
    sizeY= template.shape[0]
    img[posY:posY+sizeY, posX:posX+sizeX] = template
    return img

img_noise = np.random.randint(0,5, size = (200,200)).astype('uint8')

template=np.array([
    [255,128,128,128,255],
    [0, 255 ,0, 255,0],
    [128,128,255,128,128],
    [0,255,0,255,0],
    [255,128,128,128,255]])

#img_noise[100:105,100:105] = template
img_noise = addTemplateToIMG(img_noise, template, 31, 31)
img_noise = addTemplateToIMG(img_noise, template, 31, 51)
img_noise = addTemplateToIMG(img_noise, template, 51, 31)
img_noise = addTemplateToIMG(img_noise, template, 51, 51)

img_noise = addTemplateToIMG(img_noise, template, 151, 31)
img_noise = addTemplateToIMG(img_noise, template, 151, 51)
img_noise = addTemplateToIMG(img_noise, template, 171, 31)
img_noise = addTemplateToIMG(img_noise, template, 171, 51)

img_noise = addTemplateToIMG(img_noise, template, 98, 71)
img_noise = addTemplateToIMG(img_noise, template, 98, 81)
img_noise = addTemplateToIMG(img_noise, template, 98, 91)
img_noise = addTemplateToIMG(img_noise, template, 98, 101)
img_noise = addTemplateToIMG(img_noise, template, 98, 111)
img_noise = addTemplateToIMG(img_noise, template, 98, 121)


img_noise = addTemplateToIMG(img_noise, template, 70, 144)
img_noise = addTemplateToIMG(img_noise, template, 80, 154)
img_noise = addTemplateToIMG(img_noise, template, 90, 164)
img_noise = addTemplateToIMG(img_noise, template, 100, 174)
img_noise = addTemplateToIMG(img_noise, template, 110, 164)
img_noise = addTemplateToIMG(img_noise, template, 120, 154)
img_noise = addTemplateToIMG(img_noise, template, 130, 144)