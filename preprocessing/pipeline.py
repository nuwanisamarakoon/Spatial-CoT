import numpy as np

from preprocessing.dataset_indexer import DatasetIndexer
from preprocessing.frame_sampler import FrameSampler
from preprocessing.image_processor import ImageProcessor
from preprocessing.depth_estimator import PseudoDepthEstimator
from preprocessing.pointcloud_generator import PointCloudGenerator


class PreprocessingPipeline:
    def __init__(self, root_dir, split="train", sequence_length=8):
        self.indexer = DatasetIndexer(root_dir, split)
        self.sampler = FrameSampler(sequence_length)
        self.processor = ImageProcessor()
        self.depth_estimator = PseudoDepthEstimator()
        self.pc_generator = PointCloudGenerator()

    def process_sample(self, sample):
        image_paths = sample["images"]

        sampled_paths = self.sampler.sample(image_paths)

        rgb_seq = []
        depth_seq = []
        pointcloud_seq = []

        for path in sampled_paths:
            # Step 1: RGB processing
            rgb = self.processor.process(path)

            # Step 2: Depth estimation
            depth = self.depth_estimator.estimate(rgb)

            # Step 3: Point cloud generation
            pcd = self.pc_generator.generate(rgb, depth)

            # Step 4: Noise filtering
            pcd = self.pc_generator.filter(pcd)

            rgb_seq.append(rgb)
            depth_seq.append(depth)
            pointcloud_seq.append(pcd)

        rgb_seq = np.stack(rgb_seq)
        depth_seq = np.stack(depth_seq)

        return {
            "rgb": rgb_seq,
            "depth": depth_seq,
            "pointclouds": pointcloud_seq,
            "actions": sample["actions"],
            "meta": sample["meta"]
        }

    def __getitem__(self, idx):
        sample = self.indexer[idx]
        return self.process_sample(sample)

    def __len__(self):
        return len(self.indexer)


if __name__ == "__main__":
    pipeline = PreprocessingPipeline(
        root_dir="dataset/no_weight_processed_stitch",
        split="train",
        sequence_length=8
    )

    sample = pipeline[0]

    print("RGB:", sample["rgb"].shape)
    print("Depth:", sample["depth"].shape)
    print("PointCloud frames:", len(sample["pointclouds"]))