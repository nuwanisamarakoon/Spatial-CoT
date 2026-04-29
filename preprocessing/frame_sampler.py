import numpy as np


class FrameSampler:
    def __init__(self, sequence_length=8):
        self.sequence_length = sequence_length

    def sample(self, image_paths):
        total = len(image_paths)

        if total < self.sequence_length:
            indices = np.linspace(0, total - 1, total).astype(int)
        else:
            indices = np.linspace(0, total - 1, self.sequence_length).astype(int)

        sampled = [image_paths[i] for i in indices]
        return sampled