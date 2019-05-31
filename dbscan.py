import time
import numpy as np
from sklearn import metrics


class DBScan:
    def __init__(self, D, eps, min_points):
        self.D = D                      # data
        self.eps = eps                  # epsilon value
        self.min_points = min_points    # min points

    def params(self):
        print("Parameters: Data: {}, epsilon: {}, min numbers of points: {}".format(
            self.D, self.eps, self.min_points))

    def scan(self, show_num_clusters=False):
        # 0 for non-visited
        # -1 for noise
        start = time.time()
        labels = [0]*len(self.D)

        # define a first cluster
        C = 0

        # loop through all the points
        for current_point in range(0, len(self.D)):
            # if the label of current_points is not 0, continue
            if not (labels[current_point] == 0):
                continue

            # Define neighbors points
            NeighborPts = self.regionQuery(self.D, current_point, self.eps)

            # if the number of points in the neighborhood is less than min points, classify as noise
            if len(NeighborPts) < self.min_points:
                labels[current_point] = -1

            else:
                # Increment Cluster number
                C += 1
                # Grow the cluster with the defined params
                self.growCluster(self.D, labels, current_point,
                                 NeighborPts, C, self.eps, self.min_points)

        if(show_num_clusters):
            print(C)
        elapsed_time_fl = (time.time() - start)
        print("Time of execution {}s".format(round(elapsed_time_fl, 2)))
        return labels

    def regionQuery(self, D, current_point, eps):
        # Define list
        neighbors = []
        # For points in D
        for Pn in range(0, len(D)):
            # if the distance of the point is below epsilon, add the point to neighbors
            # Euclidian distances np.linalg.norm(a-b)
            if np.linalg.norm(D[current_point] - D[Pn]) < eps:
                neighbors.append(Pn)

        return neighbors

    def growCluster(self, D, labels, P, NeighborPts, C, eps, min_points):
        labels[P] = C
        i = 0
        while i < len(NeighborPts):
            Pn = NeighborPts[i]
            if labels[Pn] == -1:
                labels[Pn] = C
            elif labels[Pn] == 0:
                labels[Pn] = C
                PnNeighborPts = self.regionQuery(D, Pn, eps)
                if len(PnNeighborPts) >= min_points:
                    NeighborPts = NeighborPts + PnNeighborPts
            i += 1

    def results(self, result, y):
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, result))
        print("Completeness: %0.3f" % metrics.completeness_score(y, result))
        print("V-measure: %0.3f" % metrics.v_measure_score(y, result))
        print("Adjusted Rand Index: %0.3f"
              % metrics.adjusted_rand_score(y, result))
        print('\n')
