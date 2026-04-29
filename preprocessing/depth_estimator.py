import cv2
import numpy as np


class PseudoDepthEstimator:
    def estimate(self, rgb_img):
        gray = cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        depth = gray.astype(np.float32) / 255.0

        return depth