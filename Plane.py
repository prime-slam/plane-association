import copy

import numpy as np
import open3d as o3d


class Plane:
    def __init__(self, points, equation, color):
        self.points = points
        self.equation = equation
        self.color = color

    @staticmethod
    def get_normal(points):
        c = np.mean(points, axis=0)
        A = np.array(points) - c
        eigvals, eigvects = np.linalg.eig(A.T @ A)
        min_index = np.argmin(eigvals)
        n = eigvects[:, min_index]

        d = -np.dot(n, c)
        normal = int(np.sign(d)) * n
        d *= np.sign(d)
        return np.asarray([normal[0], normal[1], normal[2], d])

    def down_sample(self, sample_rate: int):
        plane_copy = copy.deepcopy(self)
        plane_copy.points = plane_copy.points[::sample_rate]
        return plane_copy

    def voxel_down_sample(self, voxel_size: float):
        if voxel_size == 0:
            return self
        plane_copy = copy.deepcopy(self)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(plane_copy.points))
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)
        plane_copy.points = np.asarray(downsampled_pcd.points)
        return plane_copy
