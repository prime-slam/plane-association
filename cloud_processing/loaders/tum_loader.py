import numpy as np
import open3d as o3d
import cv2


def icl_depth_dir_sort_func(filename: str):
    return int(filename[:-4])


def tum_depth_dir_sort_func(filename: str):
    return filename


def depth_to_pcd_custom(
    depth_image_path: str,
    camera_intrinsics: o3d.camera.PinholeCameraIntrinsic,
    depth_scale: float,
):
    points = np.zeros((camera_intrinsics.width * camera_intrinsics.height, 3))

    column_indices = np.tile(
        np.arange(camera_intrinsics.width), (camera_intrinsics.height, 1)
    ).flatten()
    row_indices = np.transpose(
        np.tile(np.arange(camera_intrinsics.height), (camera_intrinsics.width, 1))
    ).flatten()

    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
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

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd
