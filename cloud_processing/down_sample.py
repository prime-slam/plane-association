import copy

import numpy as np
import open3d as o3d


def down_sample(pcd: o3d.geometry.PointCloud, voxel_size: float, sample_rate: int):
    colors_copy = copy.deepcopy(pcd.colors)

    def get_color(index_vector_raw):
        index_vector = np.delete(index_vector_raw, np.where(index_vector_raw == -1))
        color_vector = [colors_copy[index] for index in index_vector]
        pairs, counts = np.unique(color_vector, axis=0, return_counts=True)
        return pairs[counts.argmax()]

    if voxel_size != 0:
        pcd, indices, s = pcd.voxel_down_sample_and_trace(
            voxel_size, pcd.get_min_bound(), pcd.get_max_bound()
        )
        pcd.colors = o3d.utility.Vector3dVector(
            np.apply_along_axis(get_color, 1, indices)
        )
    pcd = pcd.uniform_down_sample(sample_rate)

    return pcd
