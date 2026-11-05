import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import draw, io, measure, morphology, segmentation

# 1. Bild mit OpenCV laden (lädt standardmäßig als BGR)
image_path = 'img/beans/many5.jpg'
img_bgr = cv2.imread(image_path)

if img_bgr is None:
  print(f'Fehler beim Laden des Bildes: {image_path}. Bitte Pfad überprüfen.')
  exit()

# 2. In den HSV-Farbraum konvertieren mit OpenCV (H: 0-180, S: 0-255, V: 0-255)
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# 3. Grünen Hintergrund mit den originalen OpenCV-Schwellen filtern
mask_green = (h >= 30) & (h <= 105) & (s >= 40) & (v >= 40)
mask_beans = ~mask_green

# Rauschen entfernen (Morphologie)
footprint = morphology.disk(1)
mask_beans = morphology.opening(mask_beans, footprint)
mask_beans = morphology.closing(mask_beans, footprint)

# 4. Distanz-Transformation und Watershed (mit skimage/scipy)
distance = ndimage.distance_transform_edt(mask_beans)
max_dist = distance.max()
sure_fg = distance > (0.4 * max_dist)
markers = measure.label(sure_fg)
labels = segmentation.watershed(-distance, markers, mask=mask_beans)

# 5. Regionen analysieren (Bohnen zählen und klassifizieren)
props = measure.regionprops(labels)

white_count = 0
red_count = 0

# Für die Darstellung mit Matplotlib konvertieren wir BGR zu RGB
output_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

for prop in props:
  # Flächenfilter
  if prop.area > 100:
    region_mask = labels == prop.label
    mean_s = np.mean(s[region_mask])
    mean_v = np.mean(v[region_mask])

    y0, x0 = prop.centroid
    radius = int(prop.equivalent_diameter / 2)

    # Klassifizierung: Weiß vs. Rot/Dunkel (im nativen OpenCV-Wertebereich)
    if mean_v > 120 and mean_s < 60:
      white_count += 1
      color_mark = (0, 255, 0)  # Grün für weiße Bohnen (RGB für Matplotlib)
    else:
      red_count += 1
      color_mark = (255, 0, 0)  # Rot für rote Bohnen (RGB für Matplotlib)

    # Dicken Kreis um die Bohne zeichnen (Schleife für Linienstärke)
    try:
      base_radius = max(int(radius + 3), 3)
      for thickness_offset in range(3):
        rr, cc = draw.circle_perimeter(
            int(y0),
            int(x0),
            base_radius + thickness_offset,
            shape=output_img.shape[:2],
        )
        for c_idx, col in enumerate(color_mark):
          output_img[rr, cc, c_idx] = col
    except Exception:
      pass

# 6. Zusammenfassung ausgeben
text_summary = f'Weisse Bohnen: {white_count} | Rote Bohnen: {red_count}'
print('--- ERGEBNIS ---')
print(text_summary)

# 7. Ergebnis speichern und anzeigen
output_filename = 'Beans_Result_Skimage.jpg'
io.imsave(output_filename, output_img)
print(f'Ergebnis wurde erfolgreich unter "{output_filename}" gespeichert.')

plt.figure(figsize=(8, 8))
plt.imshow(output_img)
plt.title(text_summary)
plt.axis('off')
plt.tight_layout()
plt.show()