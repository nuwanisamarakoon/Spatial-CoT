# geometry/normal_estimation.py

import open3d as o3d


class NormalEstimator:
    def estimate(self, pcd):
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.05,
                max_nn=30
            )
        )

        return pcd