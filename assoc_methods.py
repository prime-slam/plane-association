import math
import numpy as np
from typing import List, Optional
from numbers import Number
from Plane import Plane


def first(
    plane: Plane, prev_planes: List[Plane], limit_angle: Number, limit_distance: float
) -> Optional[Plane]:
    """
    The first association method.
    :param plane: Plane from current frame to associate
    :param prev_planes: All planes from previous frame for associating
    :param limit_angle: Threshold for angle between candidates
    :param limit_distance: Threshold for distance between candidates
    :return: Associated plane or None if it doesn't exist
    """

    def get_jaccard_index(prev):
        angle_cos = np.dot(plane.equation, prev.equation) / (
            np.linalg.norm(plane.equation) * np.linalg.norm(prev.equation)
        )
        distance = math.fabs(plane.equation[-1] - prev.equation[-1])
        if (math.fabs(angle_cos) > np.cos(limit_angle)) and distance < limit_distance:
            jaccard = len(np.intersect1d(plane.points, prev.points)) / len(
                (np.union1d(plane.points, prev.points))
            )
            return jaccard
        else:
            return np.nan

    v_jaccard = np.vectorize(get_jaccard_index)
    jaccard_indices = v_jaccard(prev_planes)

    if not np.isnan(jaccard_indices).all():
        return prev_planes[np.nanargmax(jaccard_indices)]


def second(
    plane: Plane, prev_planes: List[Plane], t_dot: float, t_dist: float
) -> Optional[Plane]:
    """
    The first association method.
    :param plane: Plane from current frame to associate
    :param prev_planes: All planes from previous frame for associating
    :param t_dot: Threshold for scalar product of normals
    :param t_dist: Threshold for the ratio of the modulus of the offsets' difference and their sum
    :return: Associated plane or None if it doesn't exist
    """
    point = plane.points[0]
    normal = plane.equation[:3]
    offset = math.fabs(np.dot(point, normal))

    def get_result(prev):
        point_prev = prev.points[0]
        normal_prev = prev.equation[:3]
        offset_prev = math.fabs(np.dot(point_prev, normal))

        f_constraint = np.dot(normal, normal_prev)
        s_constraint = math.fabs(offset - offset_prev) / (offset + offset_prev)

        if (f_constraint > t_dot) and (s_constraint < t_dist):
            result = np.linalg.norm(offset * normal - offset_prev * normal_prev)
            return result
        else:
            return np.nan

    v_get = np.vectorize(get_result)
    results = v_get(prev_planes)

    if not np.isnan(results).all():
        return prev_planes[np.nanargmin(results)]


def third(plane: Plane, prev_planes: List[Plane]) -> Plane:
    """
    The third association method.
    :param plane: Plane from current frame to associate
    :param prev_planes: All planes from previous frame for associating
    :return: Associated plane
    """

    def get_result(prev):
        diff = np.linalg.norm(plane.equation - prev.equation)
        jaccard = len(np.intersect1d(plane.points, prev.points)) / len(
            (np.union1d(plane.points, prev.points))
        )
        return diff + 1 - jaccard

    v_get = np.vectorize(get_result)
    results = v_get(prev_planes)
    return prev_planes[np.argmin(results)]
