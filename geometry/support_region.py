# geometry/support_region.py

import numpy as np
from scipy.spatial import ConvexHull


class SupportRegionEstimator:
    def compute(self, pcd):
        points = np.asarray(pcd.points)

        if len(points) < 3:
            return None

        # Project to ground plane (XY)
        xy = points[:, :2]

        try:
            hull = ConvexHull(xy)
            polygon = xy[hull.vertices]
        except:
            polygon = xy

        return polygon