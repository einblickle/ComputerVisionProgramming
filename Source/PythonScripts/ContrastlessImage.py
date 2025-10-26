import numpy as np
import cv2

image = cv2.imread(r'D:\workspace\ComputerVision\Fotos\Logo_HE.png', cv2.IMREAD_GRAYSCALE)
image_clip = np.clip(image, 125, 126)

errorFactor = 0.001
indexX = np.random.randint(0, image.shape[0], int(image.size*errorFactor))
indexY = np.random.randint(0, image.shape[1], int(image.size*errorFactor))
image_clip[indexX,indexY] = 0


errorFactor = 0.003
indexX = np.random.randint(0, image.shape[0], int(image.size*errorFactor))
indexY = np.random.randint(0, image.shape[1], int(image.size*errorFactor))
image_clip[indexX,indexY] = 255

cv2.imwrite(r'D:temp\whatIsThat.png', image_clip)
