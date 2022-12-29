import math

import numpy as np
import open3d as o3d


def icl_raw_depth_dir_sort_func(filename: str):
    return int(filename.split(".")[0].split("_")[-1])


def icl_raw_depth_to_pcd_custom(depth_image_path, intrinsics, scale):
    # Adopted from https://www.doc.ic.ac.uk/~ahanda/VaFRIC/compute3Dpositions.m
    fx, fy, cx, cy = __get_camera_params_for_frame(depth_image_path, intrinsics)

    x_matrix = np.tile(np.arange(intrinsics.width), (intrinsics.height, 1)).flatten()
    y_matrix = np.transpose(
        np.tile(np.arange(intrinsics.height), (intrinsics.width, 1))
    ).flatten()
    x_modifier = (x_matrix - cx) / fx
    y_modifier = (y_matrix - cy) / fy

    points = np.zeros((intrinsics.width * intrinsics.height, 3))

    pcd = o3d.geometry.PointCloud()
    with open(depth_image_path, "r") as input_file:
        data = input_file.read()
        depth_data = np.asarray(
            list(
                map(
                    lambda x: float(x),
                    data.split(" ")[: intrinsics.height * intrinsics.width],
                )
            )
        )
        # depth_data = depth_data.reshape((480, 640))

        points[:, 2] = (
            depth_data / np.sqrt(x_modifier**2 + y_modifier**2 + 1) / scale
        )
        points[:, 0] = x_modifier * points[:, 2]
        points[:, 1] = y_modifier * points[:, 2]

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.transform([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return pcd


def __load_camera_params_from_file(depth_image_path) -> dict:
    result = {}
    params_path = depth_image_path[:-5] + "txt"
    with open(params_path, "r") as input_file:
        for line in input_file:
            field_name_start = 0
            field_name_end = line.find(" ")
            field_name = line[field_name_start:field_name_end]
            value_start = line.find("=") + 2  # skip space after '='
            if field_name == "cam_angle":
                value_end = line.find(";")
            else:
                value_end = line.find(";") - 1
            value = line[value_start:value_end]
            result[field_name] = value

        return result


def __get_camera_params_for_frame(depth_image_path, intrinsics):
    # Adopted from https://www.doc.ic.ac.uk/~ahanda/VaFRIC/getcamK.m
    camera_params_raw = __load_camera_params_from_file(depth_image_path)
    cam_dir = np.fromstring(camera_params_raw["cam_dir"][1:-1], dtype=float, sep=",").T
    cam_right = np.fromstring(
        camera_params_raw["cam_right"][1:-1], dtype=float, sep=","
    ).T
    cam_up = np.fromstring(camera_params_raw["cam_up"][1:-1], dtype=float, sep=",").T
    focal = np.linalg.norm(cam_dir)
    aspect = np.linalg.norm(cam_right) / np.linalg.norm(cam_up)
    angle = 2 * math.atan(np.linalg.norm(cam_right) / 2 / focal)

    width = intrinsics.width
    height = intrinsics.height
    psx = 2 * focal * math.tan(0.5 * angle) / width
    psy = 2 * focal * math.tan(0.5 * angle) / aspect / height

    psx = psx / focal
    psy = psy / focal

    o_x = (width + 1) * 0.5
    o_y = (height + 1) * 0.5

    fx = 1 / psx
    fy = -1 / psy
    cx = o_x
    cy = o_y

    return fx, fy, cx, cy
