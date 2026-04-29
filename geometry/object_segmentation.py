class ObjectSegmenter:
    def segment(self, object_cloud):
        # Remove tiny clusters (noise)
        labels = object_cloud.cluster_dbscan(
            eps=0.02,
            min_points=30,
            print_progress=False
        )

        if len(labels) == 0:
            return object_cloud

        max_label = max(labels)
        if max_label < 0:
            return object_cloud

        # Keep largest cluster
        largest_cluster = max(set(labels), key=list(labels).count)

        indices = [i for i, l in enumerate(labels) if l == largest_cluster]

        return object_cloud.select_by_index(indices)