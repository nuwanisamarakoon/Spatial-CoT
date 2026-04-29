# geometry/test_geometry.py

from preprocessing.pipeline import PreprocessingPipeline
from geometry.pipeline import GeometryPipeline
import open3d as o3d


def test():
    pre = PreprocessingPipeline(
        root_dir="dataset/no_weight_processed_stitch",
        split="train",
        sequence_length=8
    )

    geo = GeometryPipeline()

    sample = pre[0]

    pcd = sample["pointclouds"][0]

    result = geo.process(pcd)

    print("Plane:", result["plane_model"])
    print("Centroid:", result["centroid"])

    o3d.visualization.draw_geometries([
        result["plane_cloud"],
        result["object_cloud"]
    ])


if __name__ == "__main__":
    test()