import math
import numpy as np
from Plane import Plane


def get_jaccard_index(plane1: Plane, plane2: Plane) -> float:
    jaccard = len(np.intersect1d(plane1.points, plane2.points)) / len(
        (np.union1d(plane1.points, plane2.points))
    )
    return jaccard


def get_angle_cos(plane1: Plane, plane2: Plane) -> float:
    normal1 = plane1.equation[:3]
    normal2 = plane2.equation[:3]
    angle_cos = np.dot(normal1, normal2) / (
        np.linalg.norm(normal1) * np.linalg.norm(normal2)
    )
    return angle_cos


def get_distance(plane1: Plane, plane2: Plane) -> float:
    d1 = plane1.equation[-1]
    d2 = plane2.equation[-1]
    return math.fabs(d1 - d2)
