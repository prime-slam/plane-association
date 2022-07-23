from statistics import mean
from typing import List
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import PCD
import time
import open3d as o3d


def read_depth_and_labeled_images(path_to_depth: str, path_to_labeled_images: str):
    depth = os.listdir(path_to_depth)
    depth.sort(key=lambda a: int(os.path.splitext(a)[0]))
    labeled_images = sorted(os.listdir(path_to_labeled_images))
    return depth, labeled_images


def quality_test(
    methods: List,
    path_to_depth: str,
    path_to_labeled_images: str,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
):
    depth, labeled_images = read_depth_and_labeled_images(
        path_to_depth, path_to_labeled_images
    )
    for method in methods:
        results_planes = []
        results_points = []
        for i in tqdm(range(len(depth) - 1)):
            planes_first = PCD.depth_to_planes(
                os.path.join(path_to_depth, depth[i]),
                intrinsics,
                os.path.join(path_to_labeled_images, labeled_images[i]),
            )
            planes_second = PCD.depth_to_planes(
                os.path.join(path_to_depth, depth[i + 1]),
                intrinsics,
                os.path.join(path_to_labeled_images, labeled_images[i + 1]),
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


def performance_test(
    methods: List,
    path_to_depth: str,
    path_to_labeled_images: str,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
):
    depth, labeled_images = read_depth_and_labeled_images(
        path_to_depth, path_to_labeled_images
    )
    x = range(0, len(depth) - 1, 10)
    for method in methods:
        results = []
        for i in tqdm(x):
            planes_first = PCD.depth_to_planes(
                os.path.join(path_to_depth, depth[i]),
                intrinsics,
                os.path.join(path_to_labeled_images, labeled_images[i]),
            )
            planes_second = PCD.depth_to_planes(
                os.path.join(path_to_depth, depth[i + 1]),
                intrinsics,
                os.path.join(path_to_labeled_images, labeled_images[i + 1]),
            )

            one_frame_results = []
            for plane in planes_second:
                for j in range(10):
                    start = time.time()
                    method(plane, planes_first)
                    end = time.time()
                    one_frame_results.append(end - start)
            results.append(mean(one_frame_results))
        plt.plot(x, results, label=method.__name__)

    plt.xlabel("Position number")
    plt.title("Plane association performance")
    plt.legend()
    plt.savefig("plane_assoc_perf.pdf")
    plt.show()
