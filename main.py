import argparse
import configparser
import os
import open3d as o3d
import experiments
from assoc_methods.jaccard_thresholded import JaccardThresholded
from assoc_methods.jaccard_weighed import JaccardWeighed

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

    # For ICL NUIM
    depth_images = os.listdir(args.path_to_depth)
    depth_images.sort(key=lambda a: int(os.path.splitext(a)[0]))

    # For TUM
    # depth_images = sorted(os.listdir(args.path_to_depth))

    labeled_images = sorted(os.listdir(args.path_to_labeled_images))
    for i in range(len(depth_images)):
        depth_images[i] = os.path.join(args.path_to_depth, depth_images[i])
        labeled_images[i] = os.path.join(args.path_to_labeled_images, labeled_images[i])

    jaccard_thresholded = JaccardThresholded()
    jaccard_weighed = JaccardWeighed()
    methods = [(jaccard_weighed, 0, 16), (jaccard_thresholded, 0, 4)]
    experiments.performance_test(methods, depth_images, labeled_images, intrinsics)
    experiments.quality_test(methods, depth_images, labeled_images, intrinsics)
