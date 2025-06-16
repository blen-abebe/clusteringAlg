import numpy as np 
import matplotlib.pyplot as plt
import hdbscan
import pyemma.msm as msm


#load the data + pair the values to create a 2D array 
tilt = np.loadtxt('Rheb_GTPtiltangles.txt')
rotation = np.loadtxt('Rheb_GTProtationangles.txt')
assert len(tilt) == len(rotation), "Tilt and rotation arrays must have the same length"

#2D array making process: each row is [tilt, rotation] 
X = np.column_stack((tilt, rotation))
print(X)
print(X.shape)

#convert tilt and rotation from degrees to radians
tilt_rad = np.deg2rad(tilt)
rotation_rad = np.deg2rad(rotation)

X_circ = np.column_stack([
    np.cos(tilt_rad), np.sin(tilt_rad),
    np.cos(rotation_rad), np.sin(rotation_rad)
])


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

#merges clusters based on distance between points, does not require you to specify the number of clusters in advance
# you can choose the number of clusters you want to find, e.g., n_clusters=3
# Convert angles to radians

step = 30
X_sub = X[::step]
tilt_sub = tilt[::step]
rotation_sub = rotation[::step]
X_circ_sub = X_circ[::step]

clusterer = hdbscan.HDBSCAN(min_cluster_size=90)
labels = clusterer.fit_predict(X_circ_sub)

plt.scatter(tilt_sub, rotation_sub, c=labels, cmap='tab10', s=10)
plt.xlabel('Tilt (deg)')
plt.ylabel('Rotation (deg)')
plt.title('HDBSCAN Clustering [Circular Features, 3 Clusters]')
plt.show()

# Print number of clusters found (excluding noise, labeled as -1)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Number of clusters found: {n_clusters}")

#pyemma markov state model
#source pyemma-env/bin/activate
#python clusteringEnv.py
