import numpy as np
import matplotlib.pyplot as plt




#----------------------------------
# Funktion zur graphischen Ausgabe
# einer Konfusionsmatrix
#----------------------------------
def plotConfusionMatrix(c,norm,ax,txt=None):
    
    format_string = "{:.0f}".format
    
    if norm:
        c = c/np.sum(c,axis=1)
        format_string = "{:.2f}".format
    
    ax.imshow(c,cmap=plt.cm.Blues)
    ax.set_xlabel("Vorhergesagte Klasse")
    ax.set_ylabel("Wahre Klasse")
    
    if txt is not None:
        ax.set_title(txt)
    else:    
        if norm:
            ax.set_title("Konfusionsmatrix (normiert)")
        else: 
            ax.set_title("Konfusionsmatrix (absolut)")

    max_c = np.max(c)
    
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if c[i,j] > 0.4*max_c:
                ax.text(i,j,format_string(c[i,j]), ha='center', va='center',color='white' )
            else:
                ax.text(i,j,format_string(c[i,j]), ha='center', va='center',color='black' )