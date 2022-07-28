import numpy as np
import open3d as o3d
import cv2
from typing import List
from plane import Plane


class Transformator:
    def __init__(
        self,
        pcd: o3d.geometry.PointCloud = None,
        voxel_size: float = 0,
        sample_rate: int = 1,
    ):
        self.pcd = pcd
        self.voxel_size = voxel_size
        self.sample_rate = sample_rate

    def __create_plane_from_pcd_indices(self, indices, color) -> Plane:
        plane_points = np.asarray(self.pcd.points)[indices]
        equation = Plane.get_normal(plane_points)
        plane = Plane(plane_points, equation, color)
        return plane

    def __get_planes_npy(self, colors_npy_path: str) -> List[Plane]:
        planes = []
        annot_of_image = np.load(colors_npy_path)
        annot_unique = np.unique(annot_of_image, axis=0)
        unique_annot_without_black = list(filter(lambda x: (x != 1), annot_unique))

        for annot_num in unique_annot_without_black:
            indices = np.where(annot_of_image == annot_num)[0]

            planes.append(
                self.__create_plane_from_pcd_indices(indices, np.asarray([0, 0, 0]))
            )

        return planes

    def __get_planes_labeled(self) -> List[Plane]:
        planes = []

        colors_unique = np.unique(self.pcd.colors, axis=0)
        unique_colors_without_black = list(
            filter(lambda x: (x != [0, 0, 0]).all(axis=0), colors_unique)
        )

        for color in unique_colors_without_black:
            indices = np.where((self.pcd.colors == color).all(axis=1))[0]

            planes.append(self.__create_plane_from_pcd_indices(indices, color))

        return planes

    def __depth_to_pcd_custom(
        self,
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
            (row_indices - intrinsics_matrix[1, 2])
            * points[:, 2]
            / intrinsics_matrix[1, 1]
        )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        self.pcd = pcd

    def __annotate(self, annotation_path: str):
        annotation_rgb = cv2.imread(annotation_path)
        colors = (
            annotation_rgb.reshape(
                (annotation_rgb.shape[0] * annotation_rgb.shape[1], 3)
            )
        ) / 255
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

    def __downsample(self):
        if self.voxel_size != 0:
            self.pcd = self.pcd.voxel_down_sample(self.voxel_size)
        self.pcd = self.pcd.uniform_down_sample(self.sample_rate)

    def get_pcd_and_planes_from_depth(
        self,
        depth_image: str,
        intrinsics: o3d.camera.PinholeCameraIntrinsic,
        path_to_labeled_image: str,
        depth_scale: int,
    ) -> List[Plane]:
        """
        Creates PointCloud from depth image,
        then extracts planes with labeled images
        :param depth_image: Depth image for PointCloud creation
        :param intrinsics: Intrinsic camera parameters
        :param path_to_labeled_image: Path to labeled images
        :param depth_scale: The depth is scaled by 1 / depth_scale
        :return: Planes from PointCloud
        """
        self.__depth_to_pcd_custom(depth_image, intrinsics, depth_scale)
        self.__annotate(path_to_labeled_image)
        self.__downsample()
        return self.__get_planes_labeled()

    def get_planes_with_npy(self, colors_npy_path: str) -> List[Plane]:
        """
        Extracts planes with colors in .npy file and ready PointCloud
        :param colors_npy_path: Path to .npy file
        :return: Planes from PointCloud
        """
        if self.pcd is None:
            raise ValueError("PointCloud must not be None")
        self.__downsample()
        return self.__get_planes_npy(colors_npy_path)
