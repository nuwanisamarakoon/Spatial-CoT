import cv2
import numpy as np


class ImageProcessor:
    def __init__(self, size=(224, 224)):
        self.size = size

    def process(self, path):
        img = cv2.imread(path)

        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.size)

        img = img.astype(np.float32) / 255.0

        return img