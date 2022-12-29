from association.assoc_methods.assoc_method import AssocMethod
from association.utils import get_angle_cos, get_distance, get_jaccard_index
from dto.plane import Plane


class JaccardWeighed(AssocMethod):
    def __init__(self, angle_weight: int = 5, jaccard_weight: int = 2):
        self.angle_weight = angle_weight
        self.jaccard_weight = jaccard_weight

    def get_result(self, prev: Plane, cur: Plane) -> float:
        """
        The angle-distance-Jaccard weighed method.
        It calculates weighed sum of angles, distances
        and Jaccard indices between planes.
        :param prev: Plane from previous frame
        :param cur: Plane from current frame
        :return: The result of the metric that is
            (1 - angle_cos) * self.angle_weight
            + distance
            + (1 - jaccard) * self.jaccard_weight
        """
        angle_cos = get_angle_cos(cur, prev)
        distance = get_distance(cur, prev)
        jaccard = get_jaccard_index(prev, cur)
        result = (
            (1 - angle_cos) * self.angle_weight
            + distance
            + (1 - jaccard) * self.jaccard_weight
        )
        return result
