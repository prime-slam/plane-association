import copy
from statistics import mean
from typing import List
from matplotlib import pyplot as plt
from tqdm import tqdm
import PCD
import time
import open3d as o3d


def quality_test(
    methods: List,
    depth_images: List[str],
    labeled_images: List[str],
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
):
    for method in methods:
        results_planes = []
        results_points = []
        for i in tqdm(range(len(depth_images) - 1)):
            planes_first = PCD.depth_to_planes(
                depth_images[i], intrinsics, labeled_images[i]
            )
            planes_second = PCD.depth_to_planes(
                depth_images[i + 1], intrinsics, labeled_images[i + 1]
            )

            # Plane that doesn't exist on the previous frame can't be matched correctly
            for plane in planes_second:
                is_found = False
                for prev in planes_first:
                    if (prev.color == plane.color).all():
                        is_found = True
                if not is_found:
                    planes_second.remove(plane)

            associated = dict.fromkeys(planes_second)
            for plane in planes_second:
                planes_copy = copy.deepcopy(planes_first)
                assoc = method(plane, planes_copy)
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
    depth_images: List[str],
    labeled_images: List[str],
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
):
    x = range(0, len(depth_images) - 1, 10)
    for method in methods:
        results = []
        for i in tqdm(x):
            planes_first = PCD.depth_to_planes(
                depth_images[i], intrinsics, labeled_images[i]
            )
            planes_second = PCD.depth_to_planes(
                depth_images[i + 1], intrinsics, labeled_images[i + 1]
            )

            one_frame_results = []
            for plane in planes_second:
                for j in range(10):
                    planes_copy = copy.deepcopy(planes_first)
                    plane_copy = copy.deepcopy(plane)
                    start = time.time()
                    method(plane_copy, planes_copy)
                    end = time.time()
                    one_frame_results.append(end - start)
            results.append(mean(one_frame_results))
        plt.plot(x, results, label=method.__name__)

    plt.xlabel("Position number")
    plt.title("Plane association performance")
    plt.legend()
    plt.savefig("plane_assoc_perf.pdf")
    plt.show()
