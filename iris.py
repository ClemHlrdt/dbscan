import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler
import dbscan as db
import numpy as np
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
import pandas as pd
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

# Plot in 2D
df = pd.DataFrame(dict(x=X_r[:, 0], y=X_r[:, 1], label=y))
colors = {0: 'red', 1: 'blue', 2: 'green'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x',
               y='y', label=key, color=colors[key])
plt.show()


# class Best:
#     def __init__(self):
#         self.res = 0
#         self.eps = 0
#         self.min_pts = 0


# settings = Best()
# iter = 0
# for eps in np.arange(0.1, 2, 0.2):
#     for min_pts in range(3, 15):
#         iter += 1
#         print(iter)
#         myDB = db.DBScan(X, eps, min_pts)
#         labels = myDB.scan()
#         # myDB.results(labels, y)
#         current_res = metrics.adjusted_rand_score(y, labels)
#         if(settings.res < current_res):
#             settings.res = current_res
#             settings.eps = eps
#             settings.min_pts = min_pts
#         n_noise_ = list(labels).count(-1)


# eps = settings.eps
# min_pts = settings.min_pts
eps = 0.5
min_pts = 11
myDB = db.DBScan(X, eps, min_pts)
print("Eps: {}, min_pts:{}".format(eps, min_pts))
labels = myDB.scan()
n_noise_ = list(labels).count(-1)
#y = [y+1 for y in y]
myDB.results(labels, y)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
current_res = metrics.adjusted_rand_score(y, labels)
print("Number of clusters: {}".format(n_clusters_))
print("Number of points considered as noise: {} out of {}".format(
    n_noise_, len(labels)))
