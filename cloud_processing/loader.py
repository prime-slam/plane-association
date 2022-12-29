import os

import open3d as o3d

from cloud_processing.annotator import annotate, get_planes_labeled
from cloud_processing.down_sample import down_sample
from cloud_processing.loaders.icl_raw_loader import (
    icl_raw_depth_dir_sort_func,
    icl_raw_depth_to_pcd_custom,
)
from cloud_processing.loaders.tum_loader import (
    icl_depth_dir_sort_func,
    tum_depth_dir_sort_func,
    depth_to_pcd_custom,
)
from dto.plane import Plane


class Loader:
    def __init__(
        self,
        depth_path: str,
        depth_format: str,
        annot_path: str,
        intrinsics: o3d.camera.PinholeCameraIntrinsic,
        depth_scale: int,
        voxel_size: float = 0,
        sample_rate: int = 1,
    ):
        """
        Class for loading planes from raw data
        :param depth_path: Path to depth images folder
        :param depth_format: Depth format from list: ['tum', 'icl', 'icl_raw']
        :param annot_path: Path to labeled images
        :param intrinsics: Intrinsic camera parameters
        :param depth_scale: The depth is scaled by 1 / depth_scale
        :param voxel_size: Size of voxel for down sample
        :param sample_rate: Rate for down sample
        """
        self.depth_path = depth_path
        self.annot_path = annot_path
        self.depth_format = depth_format
        self.intrinsics = intrinsics
        self.depth_scale = depth_scale
        self.voxel_size = voxel_size
        self.sample_rate = sample_rate

        self.annot_images = os.listdir(annot_path)

        if depth_format in ["icl", "tum"]:
            self.depth_images = os.listdir(depth_path)
            if depth_format == "icl":
                sort_func = icl_depth_dir_sort_func
            else:
                sort_func = tum_depth_dir_sort_func
            self.depth_images.sort(key=sort_func)
            self.annot_images.sort(key=sort_func)

        elif depth_format in ["icl_raw"]:
            self.depth_images = list(
                filter(lambda x: x.endswith(".depth"), os.listdir(depth_path))
            )
            self.depth_images.sort(key=icl_raw_depth_dir_sort_func)
            self.annot_images.sort(key=icl_raw_depth_dir_sort_func)

    def get_frames_count(self) -> int:
        return len(self.depth_images)

    def set_down_sample_params(self, voxel_size: float, sample_rate: int):
        self.voxel_size = voxel_size
        self.sample_rate = sample_rate

    def get_planes_for_frame(self, frame_num: int) -> [Plane]:
        """
        Creates PointCloud from depth image, then extracts planes with labeled images
        :param frame_num: index of frame in dataset
        :return: Planes from PointCloud
        """
        depth_image_path = os.path.join(self.depth_path, self.depth_images[frame_num])
        annot_image_path = os.path.join(self.annot_path, self.annot_images[frame_num])
        if self.depth_format != "icl_raw":
            pcd = depth_to_pcd_custom(
                depth_image_path, self.intrinsics, self.depth_scale
            )
        else:
            pcd = icl_raw_depth_to_pcd_custom(
                depth_image_path, self.intrinsics, self.depth_scale
            )
        annotate(annot_image_path, pcd)
        pcd = down_sample(pcd, self.voxel_size, self.sample_rate)
        # o3d.visualization.draw_geometries([pcd])
        planes = get_planes_labeled(pcd)

        return planes
