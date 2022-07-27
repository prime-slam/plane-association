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
        x = range(len(depth_images) - 1)
        for i in tqdm(x):
            prev_planes = PCD.depth_to_planes(
                depth_images[i], intrinsics, labeled_images[i]
            )
            cur_planes = PCD.depth_to_planes(
                depth_images[i + 1], intrinsics, labeled_images[i + 1]
            )

            # Plane that doesn't exist on the previous frame must be matched with None
            non_existing_planes = set()
            for cur in cur_planes:
                is_found = False
                for prev in prev_planes:
                    if (prev.color == cur.color).all():
                        is_found = True
                        break
                if not is_found:
                    non_existing_planes.add(cur)

            associated = method(cur_planes, prev_planes)

            right = 0
            right_points = 0
            all_points = 0
            for (cur, prev) in associated.items():
                all_points += len(cur.points)
                if prev is not None:
                    if (prev.color == cur.color).all():
                        right += 1
                        right_points += len(cur.points)
                else:
                    if cur in non_existing_planes:
                        right += 1
                        right_points += len(cur.points)

            results_planes.append(right / len(associated))
            results_points.append(right_points / all_points)

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
            prev_planes = PCD.depth_to_planes(
                depth_images[i], intrinsics, labeled_images[i]
            )
            cur_planes = PCD.depth_to_planes(
                depth_images[i + 1], intrinsics, labeled_images[i + 1]
            )

            one_frame_results = []
            for _ in range(10):
                start = time.time()
                method(cur_planes, prev_planes)
                end = time.time()
                one_frame_results.append(end - start)
            results.append(mean(one_frame_results))
        plt.plot(x, results, label=method.__name__)

    plt.xlabel("Position number")
    plt.title("Plane association performance")
    plt.legend()
    plt.savefig("plane_assoc_perf.pdf")
    plt.show()
