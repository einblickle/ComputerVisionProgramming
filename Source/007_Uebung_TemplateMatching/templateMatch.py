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


def templateMatching(image, template):
    """
    Template Matching implementieren
    Input:
      image:  Gray Scale Image (2D numpy array, 8bit)
      template:  template (2D numpy array, 8bit)
    
    Output:
      result: deviation image (2D numpy array, float32)
    
    Notes:
      - Calculate the mean squared error between the template and the image patch at each position
     """
    
    # Größe des Bildes und des Templates bestimmen
    img_height, img_width = image.shape
    temp_height, temp_width = template.shape

    # fuer Berechnungen Bilder als int umwandeln
    image = image.astype(int)
    template = template.astype(int)
    
    # Ergebnis-Matrix initialisieren
    result_height = img_height - temp_height-1
    result_width = img_width - temp_width-1
    result = np.zeros((result_height, result_width), dtype=int)
    
    # Template Matching durchführen
    for y in range(result_height):
        for x in range(result_width):
            # Bildausschnitt extrahieren
            image_patch = image[y:y+temp_height, x:x+temp_width]
            # Mittlere quadratische Abweichung berechnen
            mse = np.mean((image_patch - template) ** 2)
            # Abweichung in der Ergebnis-Matrix speichern
            result[y, x] = mse
            
    return result



img_noise = np.random.randint(0,255, size = (200,200)).astype('uint8')

template=np.array([
    [255,128,0,128,255],
    [0, 255 ,0, 255,0],
    [128,128,255,128,128],
    [0,255,0,255,0],
    [255,128,0,128,255]])
template = template.astype('uint8')

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


img_noise = addTemplateToIMG(img_noise, template, 70, 162)
img_noise = addTemplateToIMG(img_noise, template, 80, 166)
img_noise = addTemplateToIMG(img_noise, template, 90, 170)
img_noise = addTemplateToIMG(img_noise, template, 100, 174)
img_noise = addTemplateToIMG(img_noise, template, 110, 170)
img_noise = addTemplateToIMG(img_noise, template, 120, 166)
img_noise = addTemplateToIMG(img_noise, template, 130, 162)


#template_match = 255*(1-cv2.matchTemplate(img_noise, template, cv2.TM_SQDIFF_NORMED))

#template_match = template_match > 220

#cv2.imwrite(r'template.bmp', template)
#cv2.imwrite(r'img_noise.bmp', img_noise)

template_match2 = templateMatching(img_noise, template)
template_match = cv2.matchTemplate(img_noise, template, cv2.TM_SQDIFF)

mask = template_match2<10