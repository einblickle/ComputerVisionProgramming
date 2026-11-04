import matplotlib.pyplot as plt
import numpy as np
import skimage
import scipy
from skimage.morphology import disk, erosion, reconstruction

# 1. Ein künstliches Test-Binärbild (Maske) erstellen
# Wir zeichnen zwei Kreise, die durch eine dünne Brücke verbunden sind
mask = np.zeros((100, 200), dtype=bool)
# Linker Kreis
yy, xx = np.ogrid[:100, :200]
mask[(xx - 60) ** 2 + (yy - 50) ** 2 < 30** 2] = True
# Rechter Kreis
mask[(xx - 140) ** 2 + (yy - 50) ** 2 < 30** 2] = True
# Brücke dazwischen
mask[45:55, 60:140] = True

# 2. Marker erstellen (durch starke Erosion wird die Brücke gebrochen)
# Ein Disk-Kernel mit Radius 8 ist groß genug, um die 10 Pixel dicke Brücke zu kappen
kernel = disk(8)
marker = erosion(mask, footprint=kernel)

# 3. Morphologische Rekonstruktion (Opening by Reconstruction)
# Methode 'dilation' lässt den Marker innerhalb der Maske wachsen
reconstructed = reconstruction(marker, mask, method='dilation')

# 4. Ergebnisse visualisieren
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(mask, cmap='gray')
axes[0].set_title('1. Maske (Original mit Brücke)')
axes[0].axis('off')

axes[1].imshow(marker, cmap='gray')
axes[1].set_title('2. Marker (Nach Erosion, Brücke weg)')
axes[1].axis('off')

axes[2].imshow(reconstructed, cmap='gray')
axes[2].set_title('3. Rekonstruiertes Ergebnis')
axes[2].axis('off')

plt.tight_layout()
plt.show()

distance = scipy.ndimage.distance_transform_edt(marker > 0)

# 2. Lokale Maxima (Kerne der getrennten Objekte) finden
coords = skimage.feature.peak_local_max(distance, footprint=np.ones((3, 3)), min_distance=15)
markers_mask = np.zeros(distance.shape, dtype=bool)
markers_mask[tuple(coords.T)] = True
markers, _ = scipy.ndimage.label(markers_mask)

# 3. Watershed anwenden, um die Brücke zu schneiden
labels = skimage.segmentation.watershed(-distance, markers, mask=(marker > 0))