import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# Create raw data
samples = np.random.normal(110, 40, 3000)


# truncate data
samples[samples<0] = 0
samples[samples>255] = 255


# histogram
hist, edges = np.histogram(samples, 25, [0,255], density=True)

binWidth = edges[0]+edges[1]
# shift to centers
edges_cent = edges+binWidth/2
edges_cent = edges_cent[:-1]


####
# cummulative Histogram
####
cumHist = np.cumsum(hist)*binWidth

####
plt.figure()
plt.plot(samples, 'bo')
plt.xlabel('sample')
plt.ylabel('value')
plt.show()

plt.figure()
plt.plot(samples, 'bo')
plt.xlabel('sample')
plt.ylabel('value')
# Add horizontal lines
for y_pos in edges:
    plt.axhline(y=y_pos, color='r', linestyle='--', linewidth=1.5)
plt.show()


plt.figure()
plt.xlabel('value')
plt.ylabel('probability')
plt.bar(edges_cent, hist, width=binWidth * 0.9, color='red', edgecolor='black', alpha=0.7)
plt.show()


fig, ax = plt.subplots(1, 2)
ax[0].plot(samples, 'bo')
ax[0].set_xlabel('sample')
ax[0].set_ylabel('value')
# Add horizontal lines
for y_pos in edges:
    ax[0].axhline(y=y_pos, color='r', linestyle='--', linewidth=1.5)

ax[1].set_ylabel('value')
ax[1].set_xlabel('probability')
ax[1].barh( edges_cent, hist, height=binWidth * 0.9, edgecolor='black', alpha=0.7 )



plt.show()



plt.figure()
plt.xlabel('value')
plt.ylabel('cummulated probability')
#plt.bar(edges_cent, cumHist, width=binWidth * 0.9, color='red', edgecolor='black', alpha=0.7)
plt.plot(edges_cent, cumHist, color='blue', linestyle='-', marker='o', alpha=0.7)
plt.ylim(-0.05,1.05)
plt.xlim(0,255)
plt.axhline(0)
plt.show()
