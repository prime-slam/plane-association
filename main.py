import argparse
import configparser
import open3d as o3d
import experiments
from association.assoc_methods.jaccard_thresholded import JaccardThresholded
from association.assoc_methods.jaccard_weighed import JaccardWeighed
from cloud_processing.loader import Loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_depth")
    parser.add_argument("path_to_labeled_images")
    parser.add_argument("config_path")
    parser.add_argument("depth_format")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_path)

    intrinsics_config = config["INTRINSICS"]
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        int(intrinsics_config["width"]),
        int(intrinsics_config["height"]),
        float(intrinsics_config["fx"]),
        float(intrinsics_config["fy"]),
        float(intrinsics_config["cx"]),
        float(intrinsics_config["cy"]),
    )

    scale = int(intrinsics_config["scale"])
    loader = Loader(args.path_to_depth, args.depth_format, args.path_to_labeled_images, intrinsics, scale)

    jaccard_thresholded = JaccardThresholded()
    jaccard_weighed = JaccardWeighed()
    methods = [(jaccard_weighed, 0, 1), (jaccard_thresholded, 0, 1)]

    experiments.performance_test(methods, loader)
    experiments.quality_test(methods, loader)
