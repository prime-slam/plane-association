import math
import numpy as np
from typing import List, Optional
from numbers import Number
from Plane import Plane
from utils import get_angle_cos, get_distance, get_jaccard_index


def angle_distance_jaccard(
    plane: Plane,
    prev_planes: List[Plane],
    limit_angle: Number = np.pi / 18,
    limit_distance: float = 0.1,
) -> Optional[Plane]:
    """
    The angle-distance-jaccard method. First, the method filters out planes
    with an angle or distance greater than the threshold value, then
    the plane with the maximum Jaccard index is chosen.
    :param plane: Plane from current frame to associate
    :param prev_planes: All planes from previous frame for associating
    :param limit_angle: Threshold for angle between candidates
    :param limit_distance: Threshold for distance between candidates
    :return: Associated plane or None if it doesn't exist
    """

    jaccard_indices = []
    for prev in prev_planes:
        angle_cos = get_angle_cos(plane, prev)
        distance = get_distance(plane, prev)
        if (math.fabs(angle_cos) > np.cos(limit_angle)) and distance < limit_distance:
            jaccard_indices.append(get_jaccard_index(plane, prev))
        else:
            jaccard_indices.append(np.nan)

    if not np.isnan(jaccard_indices).all():
        return prev_planes[np.nanargmax(jaccard_indices)]


def offset_normal(
    plane: Plane, prev_planes: List[Plane], t_dot: float = 0.5, t_dist: float = 2.0
) -> Optional[Plane]:
    """
    The offset-normal method. It matches planes with minimum
    (offset_a * normal_a - offset_b * normal_b) norm.
    Also, two constraints must be satisfied.
    :param plane: Plane from current frame to associate
    :param prev_planes: All planes from previous frame for associating
    :param t_dot: Threshold for scalar product of normals
    :param t_dist: Threshold for the ratio of the modulus of the offsets' difference and their sum
    :return: Associated plane or None if it doesn't exist
    """
    point = plane.points[0]
    normal = plane.equation[:3]
    offset = math.fabs(np.dot(point, normal))

    results = []
    for prev in prev_planes:
        point_prev = prev.points[0]
        normal_prev = prev.equation[:3]
        offset_prev = math.fabs(np.dot(point_prev, normal))

        f_constraint = np.dot(normal, normal_prev)
        s_constraint = math.fabs(offset - offset_prev) / (offset + offset_prev)

        if (f_constraint > t_dot) and (s_constraint < t_dist):
            result = np.linalg.norm(offset * normal - offset_prev * normal_prev)
            results.append(result)
        else:
            results.append(np.nan)

    if not np.isnan(results).all():
        return prev_planes[np.nanargmin(results)]


def norm_jaccard(plane: Plane, prev_planes: List[Plane]) -> Plane:
    """
    The equation norm - jaccard method.
    It matches with minimum (norm between planes' equations + 1 - Jaccard index).
    :param plane: Plane from current frame to associate
    :param prev_planes: All planes from previous frame for associating
    :return: Associated plane
    """

    results = []
    for prev in prev_planes:
        diff = np.linalg.norm(plane.equation - prev.equation)
        jaccard = len(np.intersect1d(plane.points, prev.points)) / len(
            (np.union1d(plane.points, prev.points))
        )
        results.append(diff + 1 - jaccard)

    return prev_planes[np.argmin(results)]
