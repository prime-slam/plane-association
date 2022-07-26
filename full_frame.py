import math
import numpy as np
from typing import List
from numbers import Number
from Plane import Plane
from utils import get_angle_cos, get_distance, get_jaccard_index


def full_frame(
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
    jaccards = []

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
        jaccards.append(jaccard_indices)

    end = False
    associated = dict.fromkeys(cur_planes)
    while not end:
        indices_max = []
        for jaccard_list in jaccards:
            if not np.isnan(jaccard_list).all():
                indices_max.append(np.nanargmax(jaccard_list))
            else:
                indices_max.append(np.nan)

        unique_indices, counts = np.unique(indices_max, return_counts=True)
        dup = unique_indices[counts > 1]
        dup = dup[~np.isnan(dup)]
        if dup.size > 0:
            repeat = np.where(indices_max == dup[0])
            f_repeat, s_repeat = repeat[0][0], repeat[0][1]
            jaccards_1 = jaccards[f_repeat]
            jaccards_2 = jaccards[s_repeat]
            if np.nanmax(jaccards_1) > np.nanmax(jaccards_2):
                jaccards[s_repeat][np.nanargmax(jaccards[s_repeat])] = np.nan
            else:
                jaccards[f_repeat][np.nanargmax(jaccards[f_repeat])] = np.nan
        else:
            i = 0
            for cur in cur_planes:
                if indices_max[i] is np.nan:
                    associated[cur] = None
                else:
                    associated[cur] = prev_planes[indices_max[i]]
                i += 1
            end = True

    return associated
