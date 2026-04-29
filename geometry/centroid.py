# geometry/centroid.py

import numpy as np


class CentroidEstimator:
    def compute(self, pcd):
        points = np.asarray(pcd.points)

        if len(points) == 0:
            return np.array([0, 0, 0])

        centroid = points.mean(axis=0)

        return centroid