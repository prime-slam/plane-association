import numpy as np


class Plane:
    def __init__(self, points, equation, color):
        self.points = points
        self.equation = equation
        self.color = color

    @staticmethod
    def get_normal(points):
        c = np.mean(points, axis=0)
        A = np.array(points) - c
        eigvals, eigvects = np.linalg.eig(A.T @ A)
        min_index = np.argmin(eigvals)
        n = eigvects[:, min_index]

        d = -np.dot(n, c)
        normal = int(np.sign(d)) * n
        d *= np.sign(d)
        return np.asarray([normal[0], normal[1], normal[2], d])

    def down_sample(self, sample_rate: int):
        self.points = self.points[::sample_rate]
