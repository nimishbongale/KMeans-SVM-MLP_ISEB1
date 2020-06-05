#fetch dataset
!pip install RegscorePy
!wget https://raw.githubusercontent.com/MSPawanRanjith/FileTransfer/master/kmean_dataset.csv

# Commented out IPython magic to ensure Python compatibility.
#import libraries

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from sklearn import mixture
from sklearn.cluster import KMeans
from math import log

#read data, convert to dataframe,numpy array

data=pd.read_csv('kmean_dataset.csv')
X=data.to_numpy()

fig = px.scatter_3d(X, x=0, y=1, z=2)
fig.show()

#Not clear what the elbow point could be, rough plot for 2 clusters

kmeans = cluster.KMeans(n_clusters=2, n_jobs=-1)
kmeans.fit(X)
p_label = kmeans.labels_

data['p_label'] = p_label
XY = data.to_numpy()
fig = px.scatter_3d(XY, x=0, y=1, z=2, color=3)
fig.show()

#plot loss function to find elbow point (rough estimation)

loss = []
ks = range(1, 10, 1)
for k in ks:
    kmeans = cluster.KMeans(n_clusters=k, n_jobs=1)
    kmeans.fit(X)
    loss.append(kmeans.inertia_)
    
plt.plot(ks, loss, 'x-')
plt.xlabel("K")
plt.ylabel("KMeans loss function")
plt.show()

def compute_bic(kmeans,X):
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    m = kmeans.n_clusters
    n = np.bincount(labels)
    N, d = X.shape

    # using euclidean distance
    cl_var = (1.0 / (N - m) / d) * sum([sum(scipy.spatial.distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')) for i in range(m)])

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) -  0.5 * m * np.log(N) * (d+1)

    return(BIC)



from sklearn import datasets,cluster
import scipy
data=pd.read_csv('kmean_dataset.csv')
X = data.to_numpy()
Y = p_label

ks = range(1,10)

KMeans = [cluster.KMeans(n_clusters = i, init="k-means++").fit(X) for i in ks]
BIC = [compute_bic(kmeansi,X) for kmeansi in KMeans]

print(BIC)

import matplotlib.pyplot as plt
plt.plot(range(2,11),BIC)
plt.ylabel('BIC score')
plt.xlabel('Number of clusters K')
plt.show()

# As 3 was obtained as the best number for K, we split into 3 clusters
kmeans = cluster.KMeans(n_clusters=3, n_jobs=-1)
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

The optimum number of clusters were found to be 3 using Bayesian Information Criterion. Other K values may also suffice needs depending on the Covariance Types.
"""