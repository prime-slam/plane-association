import csv
import os.path

from matplotlib import pyplot as plt

if __name__ == "__main__":
    quality = [
        "plane_assoc_planes_icl_living.csv",
        "plane_assoc_points_icl_living.csv",
        "plane_assoc_planes_icl_office.csv",
        "plane_assoc_points_icl_office.csv",
        "plane_assoc_planes_tum_cabinet.csv",
        "plane_assoc_points_tum_cabinet.csv",
        "plane_assoc_planes_tum_desk.csv",
        "plane_assoc_points_tum_desk.csv",
        "plane_assoc_planes_tum_long_office.csv",
        "plane_assoc_points_tum_long_office.csv",
        "plane_assoc_planes_tum_pioneer.csv",
        "plane_assoc_points_tum_pioneer.csv",
    ]
    for filename in quality:
        filepath = os.path.join("done", filename)
        results = {}
        with open(filepath, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",", quotechar="|")

            for i, row in enumerate(reader):
                if i == 0:
                    algo_names = list(map(lambda x: x.split("_")[0], row))
                    algo_names = [
                        name.replace("Jaccard", "IoU").replace("Weighed", "Weighted")
                        for name in algo_names
                    ]
                    for name in algo_names:
                        results[name] = []
                    continue

                for index, value in enumerate(row):
                    results[algo_names[index]].append(float(value))

        min_y = 1.0
        for algo_name in algo_names:
            min_y = min(min_y, min(results[algo_name]))

        metric_type = filename.split("_")[2]
        dataset_name = "_".join(filename.split("_")[3:])[:-4]
        for algo_name in algo_names:
            x = range(0, i * 10, 10)
            plt.plot(x, results[algo_name], label=f"{algo_name}")
            plt.xlabel("Frame number")
            delta = max((1 - min_y) / 10, 0.005)
            plt.ylim([min_y - delta, 1.0 + delta])
            plt.title(f"{algo_name}, {dataset_name}, {metric_type} score")
            plt.savefig(
                os.path.join(
                    "experiment_results",
                    f"assoc_{algo_name}_{metric_type}_{dataset_name}.png",
                )
            )
            plt.show()

    performance = [
        "plane_assoc_perf_icl_living.csv",
        "plane_assoc_perf_icl_office.csv",
        "plane_assoc_perf_tum_cabinet.csv",
        "plane_assoc_perf_tum_desk.csv",
        "plane_assoc_perf_tum_long_office.csv",
        "plane_assoc_perf_tum_pioneer.csv",
    ]
    for filename in performance:
        filepath = os.path.join("done", filename)
        results = {}
        with open(filepath, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",", quotechar="|")

            for i, row in enumerate(reader):
                if i == 0:
                    algo_names = list(map(lambda x: x.split("_")[0], row))
                    algo_names = [
                        name.replace("Jaccard", "IoU").replace("Weighed", "Weighted")
                        for name in algo_names
                    ]
                    for name in algo_names:
                        results[name] = []
                    continue

                for index, value in enumerate(row):
                    results[algo_names[index]].append(float(value))

        metric_type = filename.split("_")[2]
        dataset_name = "_".join(filename.split("_")[3:])[:-4]
        for algo_name in algo_names:
            x = range(0, i * 10, 10)
            plt.plot(x, results[algo_name], label=f"{algo_name}")
        plt.xlabel("Frame number")
        plt.ylabel("Frames pair association, ms")
        plt.title(f"{dataset_name}, performance")
        plt.legend()
        plt.savefig(
            os.path.join("experiment_results", f"assoc_perf_{dataset_name}.png")
        )
        plt.show()
