import numpy as np
import skimage
import cv2

coins = skimage.data.coins()
coins = cv2.imread(r'images/coins.jpg', cv2.IMREAD_GRAYSCALE)

coinsBlur = cv2.medianBlur(coins, 11)

sobX = cv2.Sobel(coins,cv2.CV_16SC1, 1, 0, 3)
sobY = cv2.Sobel(coins,cv2.CV_16SC1, 0, 1, 3)


edges = cv2.Canny(coinsBlur, 200,70, 7)

Accum = skimage.transform.hough_circle(edges, 50)

max_index = np.argmax(Accum)
max_coordinates = np.unravel_index(max_index, Accum.shape)
