# geometry/pipeline.py

from geometry.plane_detection import PlaneDetector
from geometry.object_segmentation import ObjectSegmenter
from geometry.normal_estimation import NormalEstimator
from geometry.centroid import CentroidEstimator
from geometry.support_region import SupportRegionEstimator


class GeometryPipeline:
    def __init__(self):
        self.plane_detector = PlaneDetector()
        self.segmenter = ObjectSegmenter()
        self.normal_estimator = NormalEstimator()
        self.centroid_estimator = CentroidEstimator()
        self.support_estimator = SupportRegionEstimator()

    def process(self, pcd):
        plane_model, plane_cloud, object_cloud = self.plane_detector.detect(pcd)

        object_cloud = self.segmenter.segment(object_cloud)

        object_cloud = self.normal_estimator.estimate(object_cloud)

        centroid = self.centroid_estimator.compute(object_cloud)

        support_polygon = self.support_estimator.compute(object_cloud)

        return {
            "plane_model": plane_model,
            "plane_cloud": plane_cloud,
            "object_cloud": object_cloud,
            "centroid": centroid,
            "support_polygon": support_polygon
        }