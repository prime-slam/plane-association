import argparse
import configparser
import os
from functools import partial

import open3d as o3d
import experiments
from assoc_methods import angle_distance_jaccard, angle_distance_jaccard_weighed

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

    angle_distance_jaccard_weighed_downsampled2 = partial(
        angle_distance_jaccard_weighed, down_sample=2
    )
    angle_distance_jaccard_weighed_downsampled2.__name__ = "weighed_d2"
    angle_distance_jaccard_weighed_downsampled3 = partial(
        angle_distance_jaccard_weighed, down_sample=3
    )
    angle_distance_jaccard_weighed_downsampled3.__name__ = "weighed_d3"
    angle_distance_jaccard_weighed_downsampled4 = partial(
        angle_distance_jaccard_weighed, down_sample=4
    )
    angle_distance_jaccard_weighed_downsampled4.__name__ = "weighed_d4"

    # For ICL NUIM
    # depth_images = os.listdir(args.path_to_depth)
    # depth_images.sort(key=lambda a: int(os.path.splitext(a)[0]))

    # For TUM
    depth_images = sorted(os.listdir(args.path_to_depth))

    labeled_images = sorted(os.listdir(args.path_to_labeled_images))
    for i in range(len(depth_images)):
        depth_images[i] = os.path.join(args.path_to_depth, depth_images[i])
        labeled_images[i] = os.path.join(args.path_to_labeled_images, labeled_images[i])

    methods = [
        angle_distance_jaccard,
        angle_distance_jaccard_weighed,
        angle_distance_jaccard_weighed_downsampled2,
        angle_distance_jaccard_weighed_downsampled3,
        angle_distance_jaccard_weighed_downsampled4,
    ]
    experiments.performance_test(methods, depth_images, labeled_images, intrinsics)
    experiments.quality_test(methods, depth_images, labeled_images, intrinsics)
