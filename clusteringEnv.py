import numpy as np 
import matplotlib.pyplot as plt
import hdbscan
import pyemma
import pyemma.msm as msm
import pyemma.coordinates as coor


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


# remove noise points (labeled -1) and keep only valid clustered points
valid_idx = labels != -1
clustered_traj = labels[valid_idx].astype(int)  

# Lag times to test (in frames/steps)
lag_times = np.arange(1, 101, 5)

# Calculate implied timescales
its = pyemma.msm.its([clustered_traj], lags=lag_times, n_jobs=1)

# Plot implied timescales
pyemma.plots.plot_implied_timescales(its, units='steps', dt=1)
plt.title('Implied Timescales vs Lag Time')
plt.show()
#red is the second slowest process, blue is the slowest process 
#lag is around 30 steps 
chosen_lag = 30

msm_model = msm.estimate_markov_model([clustered_traj], lag=chosen_lag)
print("\nTransition Matrix:", msm_model.transition_matrix)
print("\nStationary distribution:", msm_model.stationary_distribution)
print("\nTimescales (steps):", msm_model.timescales(k=3)) 

# Run CK test for 3 metastable states (since you have 3 clusters)
cktest = msm_model.cktest(3)

# Plot CK test results with improved layout
plt.figure(figsize=(10, 10))  
pyemma.plots.plot_cktest(cktest)
plt.tight_layout()           
plt.show()

# use stride to reduce data size
stride = 30
X_kmeans = X[::stride]

# Perform k-means clustering (choose k=3 for 3 clusters/metastable states)
k = 3
cluster = coor.cluster_kmeans(X_kmeans, k=k, max_iter=100, stride=1, fixed_seed=42)

# assign all data to clusters (discrete trajectories)
dtrajs = cluster.assign(X)[0]  # returns a list, take the first element

# Plot cluster centers
plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.3, label='data')
plt.scatter(cluster.clustercenters[:, 0], cluster.clustercenters[:, 1], c='red', s=100, label='centers')
plt.xlabel('Tilt')
plt.ylabel('Rotation')
plt.title('PyEMMA k-means Clustering')
plt.legend()
plt.show()

