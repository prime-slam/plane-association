from typing import List

import cv2
import numpy as np
import open3d as o3d

from dto.plane import Plane


def get_planes_labeled(pcd: o3d.geometry.PointCloud) -> List[Plane]:
    planes = []

    colors_unique = np.unique(pcd.colors, axis=0)
    unique_colors_without_black = list(
        filter(lambda x: (x != [0, 0, 0]).all(axis=0), colors_unique)
    )

    for color in unique_colors_without_black:
        indices = np.where((pcd.colors == color).all(axis=1))[0]
        plane_points = np.asarray(pcd.points)[indices]
        equation = Plane.get_normal(plane_points)
        planes.append(Plane(plane_points, indices, equation, color))

    return planes


def annotate(annotation_path: str, pcd: o3d.geometry.PointCloud):
    if annotation_path.endswith(".npy"):
        __annotate_with_npy(annotation_path, pcd)
    else:
        __annotate_with_rgb(annotation_path, pcd)


def __annotate_with_rgb(annotation_path: str, pcd: o3d.geometry.PointCloud):
    annotation_rgb = cv2.imread(annotation_path)
    colors = (
        annotation_rgb.reshape((annotation_rgb.shape[0] * annotation_rgb.shape[1], 3))
    ) / 255
    pcd.colors = o3d.utility.Vector3dVector(colors)


def __annotate_with_npy(annotation_path: str, pcd: o3d.geometry.PointCloud):
    annot_of_image = np.load(annotation_path)
    annot_unique = np.unique(annot_of_image, axis=0)
    colors = np.empty((len(pcd.points), 3))
    unique_colors = {(0, 0, 0)}
    for annot_num in annot_unique:
        indices = np.where(annot_of_image == annot_num)[0]
        if annot_num == 1:
            colors[indices] = [0, 0, 0]
            continue
        unique_color = np.random.random(3)
        while tuple(unique_color) in unique_colors:
            unique_color = np.random.random(3)
        colors[indices] = unique_color
        unique_colors.add(tuple(unique_color))
    pcd.colors = o3d.utility.Vector3dVector(colors)
