import csv
from statistics import mean
from typing import List, Tuple
from matplotlib import pyplot as plt
from tqdm import tqdm

import time

from association.assoc_methods.assoc_method import AssocMethod
from association.associator import Associator

from cloud_processing.loader import Loader


def quality_test(
    methods: List[Tuple[AssocMethod, float, int]],
    loader: Loader
):
    def plot_metric_res(x, method_res, metric_type: str, method_name: str, min_y: float):
        plt.plot(x, method_res, label=method_name)

        plt.xlabel("Position number")
        plt.title(
            f"{method_name}, {metric_type} score"
        )
        plt.savefig(
            f"{method_name}_plane_assoc_{metric_type}.pdf"
        )
        delta = max((1 - min_y) / 10, 0.005)
        plt.ylim([min_y - delta, 1. + delta])
        plt.show()

    algo_plane_results = {}
    algo_point_results = {}
    plane_min_y = 1
    point_min_y = 1
    x = range(0, loader.get_frames_count() - 1, 10)
    for method, voxel_size, sample_rate in methods:
        loader.set_down_sample_params(voxel_size, sample_rate)
        results_planes = []
        results_points = []
        for i in tqdm(x):
            prev_planes = loader.get_planes_for_frame(i)
            cur_planes = loader.get_planes_for_frame(i + 1)

            # Plane that doesn't exist on the previous frame must be matched with None
            non_existing_planes = set()
            for cur_plane in cur_planes:
                is_found = False
                for prev_plane in prev_planes:
                    if (prev_plane.color == cur_plane.color).all():
                        is_found = True
                        break
                if not is_found:
                    non_existing_planes.add(cur_plane)

            associator = Associator(cur_planes, prev_planes)
            associated = associator.associate(method)

            # TODO: use EVOPS
            right = 0
            right_points = 0
            all_points = 0
            for (cur_plane, prev_plane) in associated.items():
                all_points += len(cur_plane.points)
                if prev_plane is not None:
                    if (prev_plane.color == cur_plane.color).all():
                        right += 1
                        right_points += len(cur_plane.points)
                else:
                    if cur_plane in non_existing_planes:
                        right += 1
                        right_points += len(cur_plane.points)

            results_planes.append(right / len(associated))
            results_points.append(right_points / all_points)

        algo_plane_results[f"{type(method).__name__}_v{voxel_size}_u{sample_rate}"] = results_planes
        algo_point_results[f"{type(method).__name__}_v{voxel_size}_u{sample_rate}"] = results_points
        point_min_y = min(point_min_y, min(results_points))
        plane_min_y = min(plane_min_y, min(results_planes))

    for algo in algo_plane_results.keys():
        plot_metric_res(x, algo_plane_results[algo], "planes", algo, plane_min_y)
    for algo in algo_point_results.keys():
        plot_metric_res(x, algo_point_results[algo], "points", algo, point_min_y)

    with open('plane_assoc_planes.csv', 'w') as file:
        dump_res_to_csv(file, x, algo_plane_results)
    with open('plane_assoc_points.csv', 'w') as file:
        dump_res_to_csv(file, x, algo_point_results)


def performance_test(
    methods: List[Tuple[AssocMethod, float, int]],
    loader: Loader
):
    x = range(0, loader.get_frames_count() - 1, 10)
    total_results = {}
    for method, voxel_size, sample_rate in methods:
        loader.set_down_sample_params(voxel_size, sample_rate)
        results = []
        for i in tqdm(x):
            prev_planes = loader.get_planes_for_frame(i)
            cur_planes = loader.get_planes_for_frame(i + 1)

            associator = Associator(cur_planes, prev_planes)
            one_frame_results = []
            for _ in range(10):
                start = time.time()
                associator.associate(method)
                end = time.time()
                one_frame_results.append(end - start)
            results.append(mean(one_frame_results))
        total_results[f"{type(method).__name__}_v{voxel_size}_u{sample_rate}"] = results
        plt.plot(
            x, results, label=f"{type(method).__name__}_v{voxel_size}_u{sample_rate}"
        )

    plt.xlabel("Position number")
    plt.title("Plane association performance")
    plt.legend()
    plt.savefig("plane_assoc_perf.pdf")
    plt.show()

    with open('plane_assoc_perf.csv', 'w') as file:
        dump_res_to_csv(file, x, total_results)


def dump_res_to_csv(file, x, data_dict: dict):
    header = ",".join(list(data_dict.keys()))
    writer = csv.writer(file)
    writer.writerow(header)
    for i, _ in enumerate(x):
        frame_results = []
        for key in data_dict.keys():
            result_for_frame = data_dict[key][i]
            frame_results.append(result_for_frame)
        writer.writerow(",".join(str(frame_results)))
