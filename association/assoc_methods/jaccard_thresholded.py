import math
from numbers import Number
from typing import Optional

import numpy as np

from association.assoc_methods.assoc_method import AssocMethod
from association.utils import get_angle_cos, get_distance, get_jaccard_index
from dto.plane import Plane


class JaccardThresholded(AssocMethod):
    def __init__(self, limit_distance: float = 0.1, limit_angle: Number = np.pi / 18):
        self.limit_distance = limit_distance
        self.limit_angle = limit_angle

    def get_result(self, prev: Plane, cur: Plane) -> Optional[float]:
        """
        The angle-distance-Jaccard method.
        It calculates metric for planes
        with an angle and distance below than the threshold value.
        :param prev: Plane from previous frame
        :param cur: Plane from current frame
        :return: The result of the metric that is (1 - Jaccard index)
        """
        angle_cos = get_angle_cos(cur, prev)
        distance = get_distance(cur, prev)
        if (
            math.fabs(angle_cos) > np.cos(self.limit_angle)
        ) and distance < self.limit_distance:
            return 1 - get_jaccard_index(cur, prev)
