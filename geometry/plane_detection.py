import open3d as o3d


class PlaneDetector:
    def detect(self, pcd):
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.01,
            ransac_n=3,
            num_iterations=1000
        )

        plane_cloud = pcd.select_by_index(inliers)
        object_cloud = pcd.select_by_index(inliers, invert=True)

        return plane_model, plane_cloud, object_cloud