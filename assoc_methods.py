import math
import numpy as np
from typing import List
from numbers import Number
from Plane import Plane
from utils import get_angle_cos, get_distance, get_jaccard_index


def angle_distance_jaccard(
    cur_planes: List[Plane],
    prev_planes: List[Plane],
    limit_angle: Number = np.pi / 18,
    limit_distance: float = 0.1,
) -> dict:
    """
    The angle-distance-jaccard method. First, the method filters out planes
    with an angle or distance greater than the threshold value, then
    the plane with the maximum Jaccard index is chosen.
    :param cur_planes: All planes from current frame to associate
    :param prev_planes: All planes from previous frame for associating
    :param limit_angle: Threshold for angle between candidates
    :param limit_distance: Threshold for distance between candidates
    :return: Associated planes
    """
    associated = dict.fromkeys(cur_planes)
    for cur in cur_planes:
        jaccard_indices = []
        for prev in prev_planes:
            angle_cos = get_angle_cos(cur, prev)
            distance = get_distance(cur, prev)
            if (
                math.fabs(angle_cos) > np.cos(limit_angle)
            ) and distance < limit_distance:
                jaccard_indices.append(get_jaccard_index(cur, prev))
            else:
                jaccard_indices.append(np.nan)

        if not np.isnan(jaccard_indices).all():
            associated[cur] = prev_planes[np.nanargmax(jaccard_indices)]

    return associated


def angle_distance_jaccard_weighed_seq(
    cur_planes: List[Plane],
    prev_planes: List[Plane],
    angle_weight: float = 5,
    jaccard_weight: float = 2,
    voxel_size: float = 0,
    sample_rate: int = 1,
) -> dict:
    """
    The angle-distance-Jaccard method.
    It matches planes by minimizing
    ((1 - angle_cos) * angle_weight + distance + (1 - Jaccard) * jaccard_weight).
    :param cur_planes: All planes from current frame to associate
    :param prev_planes: All planes from previous frame for associating
    :param angle_weight: angle_weight in sum to be minimized
    :param jaccard_weight: jaccard_weight in sum to be minimized
    :param sample_rate: Sample rate, the selected point indices are [0, k, 2k, …]
    :param voxel_size: Voxel size to downsample into
    :return: Associated planes
    """
    associated = dict.fromkeys(cur_planes)
    for cur in cur_planes:
        cur_voxel_down_sampled = cur.voxel_down_sample(voxel_size)
        cur_down_sampled = cur_voxel_down_sampled.down_sample(sample_rate)
        results = []
        for prev in prev_planes:
            angle_cos = get_angle_cos(prev, cur)
            distance = get_distance(prev, cur)
            prev_voxel_down_sampled = prev.voxel_down_sample(voxel_size)
            prev_down_sampled = prev_voxel_down_sampled.down_sample(sample_rate)
            jaccard = get_jaccard_index(prev_down_sampled, cur_down_sampled)
            results.append(
                (1 - angle_cos) * angle_weight
                + distance
                + (1 - jaccard) * jaccard_weight
            )

        associated[cur] = prev_planes[np.argmin(results)]

    return associated


def angle_distance_jaccard_weighed(
    cur_planes: List[Plane],
    prev_planes: List[Plane],
    angle_weight: float = 5,
    jaccard_weight: float = 2,
    voxel_size: float = 0,
    sample_rate: int = 1,
) -> dict:
    """
    The angle-distance-Jaccard method.
    It matches planes by minimizing
    ((1 - angle_cos) * angle_weight + distance + (1 - Jaccard) * jaccard_weight).
    :param cur_planes: All planes from current frame to associate
    :param prev_planes: All planes from previous frame for associating
    :param angle_weight: angle_weight in sum to be minimized
    :param jaccard_weight: jaccard_weight in sum to be minimized
    :param sample_rate: Down sample rate, the selected point indices are [0, k, 2k, …]
    :param voxel_size: Voxel size to downsample into
    :return: Associated planes
    """
    results = {}

    for cur in cur_planes:
        cur_voxel_down_sampled = cur.voxel_down_sample(voxel_size)
        cur_down_sampled = cur_voxel_down_sampled.down_sample(sample_rate)
        for prev in prev_planes:
            cur_pair = (cur, prev)
            angle_cos = get_angle_cos(prev, cur)
            distance = get_distance(prev, cur)
            prev_voxel_down_sampled = prev.voxel_down_sample(voxel_size)
            prev_down_sampled = prev_voxel_down_sampled.down_sample(sample_rate)
            jaccard = get_jaccard_index(prev_down_sampled, cur_down_sampled)
            results[cur_pair] = (
                (1 - angle_cos) * angle_weight
                + distance
                + (1 - jaccard) * jaccard_weight
            )

    sorted_res = sorted(results.items(), key=lambda item: item[1])

    prev_used = set()
    associated = dict.fromkeys(cur_planes)
    for pair, result in sorted_res:
        cur, prev = pair

        if associated[cur] is not None or prev in prev_used:
            continue

        associated[cur] = prev
        prev_used.add(prev)

    return associated
