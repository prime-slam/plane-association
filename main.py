import argparse
import configparser
import os
import open3d as o3d
from matplotlib import pyplot as plt
from tqdm import tqdm

from assoc_methods import angle_distance_jaccard, offset_normal, norm_jaccard
import PCD

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_depth")
    parser.add_argument("path_to_planes")
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

    depth = os.listdir(args.path_to_depth)
    depth.sort(key=lambda a: int(os.path.splitext(a)[0]))
    cc = sorted(os.listdir(args.path_to_planes))

    methods = [angle_distance_jaccard, offset_normal, norm_jaccard]
    for method in methods:
        results_planes = []
        results_points = []
        for i in tqdm(range(len(depth) - 1)):
            planes_first = PCD.depth_to_planes(
                os.path.join(args.path_to_depth, depth[i]),
                intrinsics,
                os.path.join(args.path_to_planes, cc[i]),
            )
            planes_second = PCD.depth_to_planes(
                os.path.join(args.path_to_depth, depth[i + 1]),
                intrinsics,
                os.path.join(args.path_to_planes, cc[i + 1]),
            )

            associated = dict.fromkeys(planes_second)
            for plane in planes_second:
                assoc = method(plane, planes_first)
                associated[plane] = assoc

            right = 0
            right_points = 0
            all_points = 0
            for (k, v) in associated.items():
                all_points += len(k.points)
                if v is not None:
                    if (v.color == k.color).all():
                        right += 1
                        right_points += len(k.points)

            results_planes.append(right / len(associated))
            results_points.append(right_points / all_points)

        x = [i for i in range(len(results_planes))]
        y1 = results_planes
        y2 = results_points

        plt.plot(x, y1, label="Planes")
        plt.plot(x, y2, label="Points")

        plt.xlabel("Position number")
        plt.title("Plane association, " + method.__name__)
        plt.legend()
        plt.savefig("plane_assoc_" + method.__name__ + ".pdf")
        plt.show()
