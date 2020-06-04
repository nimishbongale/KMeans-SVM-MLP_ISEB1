#fetch dataset

!wget https://raw.githubusercontent.com/MSPawanRanjith/FileTransfer/master/kmean_dataset.csv

# Commented out IPython magic to ensure Python compatibility.
#import libraries

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from sklearn import mixture
from math import log

#read data, convert to dataframe,numpy array

data=pd.read_csv('kmean_dataset.csv')
X=data.to_numpy()

fig = px.scatter_3d(X, x=0, y=1, z=2)
fig.show()

#plot loss function to find elbow point (rough estimation)

from sklearn.cluster import KMeans

loss = []
ks = range(1, 10, 1)
for k in ks:
    kmeans = KMeans(n_clusters=k, n_jobs=1)
    kmeans.fit(X)
    loss.append(kmeans.inertia_)
    
plt.plot(ks, loss, 'x-')
plt.xlabel("K")
plt.ylabel("KMeans loss function")
plt.show()

#Not clear what the elbow point could be, rough plot for 3 clusters

kmeans = KMeans(n_clusters=3, n_jobs=-1)
kmeans.fit(X)
p_label = kmeans.labels_

# Demo of the rough result

data['p_label'] = p_label
XY = data.to_numpy()
fig = px.scatter_3d(XY, x=0, y=1, z=2, color=3)
fig.show()

#this function computes the BIC values and returns them. They are not first order derivated.
def BIC(kmeans):
    D, N, K = kmeans.cluster_centers_.shape[1], len(kmeans.labels_), kmeans.n_clusters
    num_count = np.zeros(K, dtype=int)
    for label in kmeans.labels_:
        num_count[label] += 1
    sigma_square = kmeans.inertia_ / (D * N)
    
    bic = 0
    for count in num_count:
        bic -= 2 * count * np.log(count * 1.0 / N)
    
    bic += N*D * np.log(2 * np.pi * sigma_square)
    
    bic += kmeans.inertia_ / (sigma_square)
    bic += K * (D + 1) * np.log(N)
    return bic

#using GMM from sklearn
def calculate_bic(X, ks, cov):
    bics = []
    best_k = 1
    best_bic = float('inf')
    best_kmeans = None

    for k in ks:
        gmm = mixture.GaussianMixture(n_components=k,
                                      covariance_type=cov)
        gmm.fit(X)
        bic=gmm.bic(X)
        bics.append(bic)
        if bic < best_bic:
            best_k = k
            best_bic = bic
            best_kmeans = kmeans
            
    return bics, best_k, best_bic, best_kmeans

#plot for differt covariance_types
def plotit(cov):
  ks = range(1, 10, 1)
  bics, best_k, best_bic, best_kmeans = calculate_bic(X, ks, cov)
  plt.plot(ks, bics, 'x-')
  plt.xlabel("K")
  plt.ylabel("KMeans BIC")
  plt.show()  

  print("Best number of k by BIC", best_k)

print("For Tied Covariance Type")
plotit('tied')

print("For full Covariance Type")
plotit('full')

# suggests making strictly 2 clusters

print("For Spherical Covariance Type")
plotit('spherical')

print("For Diag Covariance Type")
plotit('diag')

#find optimal BIC value
def calculate_bic_own(X, ks):
    bics = []
    best_k = 1
    best_bic = float('inf')
    best_kmeans = None

    for k in ks:
        kmeans = KMeans(n_clusters=k, n_jobs=-1)
        kmeans.fit(X)
        bic = BIC(kmeans)
        bics.append(bic)
        if bic < best_bic:
            best_k = k
            best_bic = bic
            best_kmeans = kmeans
            
    return bics, best_k, best_bic, best_kmeans

#direct BIC scores
ks = range(1, 10, 1)
bics, best_k, best_bic, best_kmeans = calculate_bic_own(X, ks)
print(bics)
plt.plot(ks, bics, 'x-')
plt.xlabel("K")
plt.ylabel("KMeans BIC (Normal Graph)")
plt.show()
print("Best number of k by BIC", best_k)

#BIC first derivative curve
ks = range(1, 10, 1)
bics, best_k, best_bic, best_kmeans = calculate_bic_own(X, ks)
bicscpy = np.gradient(np.array(bics)).tolist()
print(bicscpy)
plt.plot(ks, bicscpy, 'x-')
plt.xlabel("K")
plt.ylabel("KMeans BIC (First Order Derivative, to find max point)")
plt.show()
print("Best number of k by BIC", best_k)

# As 7 was obtained as the best number for K, we split the 
kmeans = KMeans(n_clusters=7, n_jobs=-1)
kmeans.fit(X)
p_label = kmeans.labels_

print("Predicted Labels")
print(p_label)
print("Cluster Centre Vectors")
print(kmeans.cluster_centers_)

data['p_label'] = p_label
XY = data.to_numpy()
fig = px.scatter_3d(XY, x=0, y=1, z=2, color=3)
fig.show()

"""# Conclusion

The optimum number of clusters were found to be 7 using Bayesian Information Criterion. Other K values may also suffice needs depending on the Covariance Types.
"""
