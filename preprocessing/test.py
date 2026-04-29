from preprocessing.pipeline import PreprocessingPipeline
import open3d as o3d


def test():
    pipeline = PreprocessingPipeline(
        root_dir="dataset/no_weight_processed_stitch",
        split="train",
        sequence_length=8
    )

    sample = pipeline[0]

    print("RGB:", sample["rgb"].shape)
    print("Depth:", sample["depth"].shape)
    print("PointCloud frames:", len(sample["pointclouds"]))

    # visualize first frame
    o3d.visualization.draw_geometries([sample["pointclouds"][2]])


if __name__ == "__main__":
    test()