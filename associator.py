from typing import List

from plane import Plane
from assoc_methods.assoc_method import AssocMethod
from utils import get_angle_cos, get_distance


class Associator:
    def __init__(self, cur_planes: List[Plane], prev_planes: List[Plane]):
        """
        Associates planes from current frame with planes from previous frame
        :param cur_planes: All planes from current frame to associate
        :param prev_planes: All planes from previous frame for associating
        """
        self.cur_planes = cur_planes
        self.prev_planes = prev_planes

    def associate(self, method: AssocMethod):
        """
        Matches planes by minimizing metric result without repeats
        :param method: Method for calculating metric between planes
        :return: Associated planes
        """
        results = {}
        for cur in self.cur_planes:
            for prev in self.prev_planes:
                cur_pair = (cur, prev)
                angle_cos = get_angle_cos(cur, prev)
                distance = get_distance(cur, prev)
                metric_result = method.get_result(prev, cur, angle_cos, distance)
                if metric_result is not None:
                    results[cur_pair] = metric_result

        sorted_res = sorted(results.items(), key=lambda item: item[1])

        prev_used = set()
        associated = dict.fromkeys(self.cur_planes)
        for pair, result in sorted_res:
            cur, prev = pair

            if associated[cur] is not None or prev in prev_used:
                continue

            associated[cur] = prev
            prev_used.add(prev)

        return associated
