import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import dbscan
import numpy as np
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
import pandas as pd

# Create some blobs
centers = [[1, 1], [-1, -1], [1, -1]]  # centers locations
X, y = make_blobs(n_samples=750, centers=centers,
                  cluster_std=0.4, random_state=0)


# Plot blobs
df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
colors = {0: 'red', 1: 'blue', 2: 'green'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x',
               y='y', label=key, color=colors[key])
plt.show()


# Using DBScan
print('Number of items: {}'.format(len(X)))
X = StandardScaler().fit_transform(X)
eps = 0.3
min_pts = 7

myDB = dbscan.DBScan(X, eps, min_pts)
print("Eps: {}, min_pts:{}".format(eps, min_pts))
labels = myDB.scan()
n_noise_ = list(labels).count(-1)
myDB.results(labels, y)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
current_res = metrics.adjusted_rand_score(y, labels)
print("Number of clusters: {}".format(n_clusters_))
print("Number of points considered as noise: {} out of {}".format(
    n_noise_, len(labels)))
