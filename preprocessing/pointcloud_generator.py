import numpy as np
import open3d as o3d


class PointCloudGenerator:
    def __init__(self, width=224, height=224):
        fx = fy = 200.0
        cx = width / 2
        cy = height / 2

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )

    def generate(self, rgb, depth):
        rgb_o3d = o3d.geometry.Image((rgb * 255).astype(np.uint8))

        # Scale pseudo-depth (important)
        depth_scaled = depth * 2.0
        depth_o3d = o3d.geometry.Image(depth_scaled.astype(np.float32))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1.0,
            convert_rgb_to_intensity=False
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            self.intrinsic
        )

        return pcd

    def filter(self, pcd):
        # Downsample
        pcd = pcd.voxel_down_sample(voxel_size=0.01)

        # Remove noise
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0
        )

        return pcd