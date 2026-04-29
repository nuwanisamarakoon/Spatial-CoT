import os
import glob
import numpy as np


class DatasetIndexer:
    def __init__(self, root_dir, split="train"):
        self.root_dir = os.path.join(root_dir, split)
        self.samples = self._index_dataset()

    def _index_dataset(self):
        samples = []

        for obj_id in os.listdir(self.root_dir):
            obj_path = os.path.join(self.root_dir, obj_id)

            if not os.path.isdir(obj_path):
                continue

            for motion in os.listdir(obj_path):
                motion_path = os.path.join(obj_path, motion)

                image_paths = sorted(glob.glob(os.path.join(motion_path, "*.png")))

                if len(image_paths) == 0:
                    continue

                actions_path = os.path.join(motion_path, "actions.npy")
                actions = np.load(actions_path) if os.path.exists(actions_path) else None

                samples.append({
                    "images": image_paths,
                    "actions": actions,
                    "meta": f"{obj_id}/{motion}"
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
indexer = DatasetIndexer("dataset/no_weight_processed_stitch", "train")
print(len(indexer))
print(indexer[0])