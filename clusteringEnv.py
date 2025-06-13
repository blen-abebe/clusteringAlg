import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

#load the data + pair the values to create a 2D array 
tilt = np.loadtxt('Rheb_GTPtiltangles.txt')
rotation = np.loadtxt('Rheb_GTProtationangles.txt')
assert len(tilt) == len(rotation), "Tilt and rotation arrays must have the same length"

#2D array making process: each row is [tilt, rotation] 
X = np.column_stack((tilt, rotation))
print(X)
print(X.shape)

#plot data to see how it looks 
angles = np.loadtxt('Rheb_GTProtationangles.txt') 
plt.hist(angles, bins=72, range=(-180, 180)) #5º bins from -180 to 180(covers all rotation choice)
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
plt.title('Histogram of Rheb Rotation Angles')
plt.show()

#plots a scatter plot to visualize the relationship between tilt and rotation angles
plt.scatter(tilt, rotation, s=10)
plt.xlabel('Tilt (deg)')
plt.ylabel('Rotation (deg)')
plt.title('Tilt vs Rotation')
plt.show()

#DBSCAN (density-based clustering alg) --> lables points that don't belong to any cluster as -1, can find any shaped clusters + does not require you to speciy the num in advance 
# distance from each point to its k-th neigbour in data set
k = 10
neigh = NearestNeighbors(n_neighbors=k)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

# Sort the distances to the k-th nearest neighbor
k_distances = np.sort(distances[:, k-1])

plt.figure()
plt.plot(k_distances)
plt.xlabel('Points sorted by distance')
plt.ylabel(f'{k}th nearest neighbor distance')
plt.title('k-distance plot for DBSCAN eps selection')
plt.show() # sharp change occurs around 1.5-2 so that will be the eps valye ir the neighborhood radius for clustering 

#eps --> how close points need to be to be considered the cluster (radius) 
#min samples --> min num of points to form a dense region  
db = DBSCAN(eps=2, min_samples=10).fit(X)
labels = db.labels_
plt.scatter(tilt, rotation, c=labels, cmap='tab10', s=10)
plt.xlabel('Tilt (deg)')
plt.ylabel('Rotation (deg)')
plt.title('DBSCAN Clustering (Circular Features)')
plt.show()

#num of clusters, -1 rep noise points
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Number of clusters found: {n_clusters}") #11 clusters found