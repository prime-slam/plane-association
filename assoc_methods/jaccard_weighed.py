from plane import Plane
from assoc_methods.assoc_method import AssocMethod
from utils import get_jaccard_index


class JaccardWeighed(AssocMethod):
    def __init__(self, angle_weight: int = 5, jaccard_weight: int = 2):
        self.angle_weight = angle_weight
        self.jaccard_weight = jaccard_weight

    def get_result(
        self, prev: Plane, cur: Plane, angle_cos: float, distance: float
    ) -> float:
        """
        The angle-distance-Jaccard weighed method.
        It calculates weighed sum of angles, distances
        and Jaccard indices between planes.
        :param prev: Plane from previous frame
        :param cur: Plane from current frame
        :param angle_cos: Angle between planes
        :param distance: Distance between plane
        :return: The result of the metric that is
            (1 - angle_cos) * self.angle_weight
            + distance
            + (1 - jaccard) * self.jaccard_weight
        """
        jaccard = get_jaccard_index(prev, cur)
        result = (
            (1 - angle_cos) * self.angle_weight
            + distance
            + (1 - jaccard) * self.jaccard_weight
        )
        return result
