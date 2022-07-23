import argparse
import configparser
import open3d as o3d
import experiments
from assoc_methods import angle_distance_jaccard, offset_normal, norm_jaccard

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_depth")
    parser.add_argument("path_to_labeled_images")
    parser.add_argument("config_path")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_path)

    intrinsics = config["INTRINSICS"]
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        int(intrinsics["width"]),
        int(intrinsics["height"]),
        float(intrinsics["fx"]),
        float(intrinsics["fy"]),
        float(intrinsics["cx"]),
        float(intrinsics["cy"]),
    )

    methods = [angle_distance_jaccard, offset_normal, norm_jaccard]
    experiments.quality_test(
        methods, args.path_to_depth, args.path_to_labeled_images, intrinsics
    )
