import numpy as np
import open3d as o3d
import cv2
from typing import List
from Plane import Plane


def get_planes(pcd: o3d.geometry.PointCloud) -> List[Plane]:
    planes_of_image = []

    colors_unique = np.unique(pcd.colors, axis=0)
    unique_colors_without_black = list(
        filter(lambda x: (x != [0, 0, 0]).all(axis=0), colors_unique)
    )

    for color in unique_colors_without_black:
        indices = np.where((pcd.colors == color).all(axis=1))[0]
        plane_points = np.asarray(pcd.points)[indices]
        equation = Plane.get_normal(plane_points)
        plane = Plane(plane_points, equation, color)
        planes_of_image.append(plane)

    return planes_of_image


def depth_to_pcd_custom(
    depth_image: np.array,
    camera_intrinsics: o3d.camera.PinholeCameraIntrinsic,
    depth_scale: float,
) -> (o3d.geometry.PointCloud, np.array):
    points = np.zeros((camera_intrinsics.width * camera_intrinsics.height, 3))

    column_indices = np.tile(
        np.arange(camera_intrinsics.width), (camera_intrinsics.height, 1)
    ).flatten()
    row_indices = np.transpose(
        np.tile(np.arange(camera_intrinsics.height), (camera_intrinsics.width, 1))
    ).flatten()

    points[:, 2] = depth_image.flatten() / depth_scale
    intrinsics_matrix = camera_intrinsics.intrinsic_matrix
    points[:, 0] = (
        (column_indices - intrinsics_matrix[0, 2])
        * points[:, 2]
        / intrinsics_matrix[0, 0]
    )
    points[:, 1] = (
        (row_indices - intrinsics_matrix[1, 2]) * points[:, 2] / intrinsics_matrix[1, 1]
    )

    zero_depth_indices = np.where(points[:, 2] == 0)[0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd, zero_depth_indices


def annotate(
    pcd: o3d.geometry.PointCloud, annotation_path: str
) -> o3d.geometry.PointCloud:
    annotation_rgb = cv2.imread(annotation_path)
    colors = (
        annotation_rgb.reshape((annotation_rgb.shape[0] * annotation_rgb.shape[1], 3))
    ) / 255
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def depth_to_planes(
    depth_image: str, intrinsics: o3d.camera.PinholeCameraIntrinsic, image_colors: str
) -> List[Plane]:
    image = cv2.imread(depth_image, cv2.IMREAD_ANYDEPTH)
    pcd, _ = depth_to_pcd_custom(image, intrinsics, 5000)
    pcd = annotate(pcd, image_colors)
    planes = get_planes(pcd)
    return planes
