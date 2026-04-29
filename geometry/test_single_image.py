import cv2
import numpy as np
import open3d as o3d

from preprocessing.image_processor import ImageProcessor
from preprocessing.depth_estimator import PseudoDepthEstimator
from preprocessing.pointcloud_generator import PointCloudGenerator
from geometry.pipeline import GeometryPipeline


def test_single_image(image_path):
    # Step 1: preprocess image
    processor = ImageProcessor()
    rgb = processor.process(image_path)

    # Step 2: pseudo depth
    depth_estimator = PseudoDepthEstimator()
    depth = depth_estimator.estimate(rgb)

    # Step 3: point cloud
    pc_gen = PointCloudGenerator()
    pcd = pc_gen.generate(rgb, depth)
    pcd = pc_gen.filter(pcd)

    # Step 4: geometry
    geo = GeometryPipeline()
    result = geo.process(pcd)

    print("Centroid:", result["centroid"])
    print("Plane Model:", result["plane_model"])

    # Visualize
    o3d.visualization.draw_geometries([
        result["object_cloud"]
    ])


if __name__ == "__main__":
    test_single_image("dataset/ball.jpeg")