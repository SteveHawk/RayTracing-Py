import numpy as np  # type: ignore
from utils.vec3 import Vec3, Point3


class Ray:
    def __init__(self, origin: Point3 = Point3(),
                 direction: Vec3 = Vec3()) -> None:
        self.orig = origin
        self.dir = direction

    def origin(self) -> Point3:
        return self.orig

    def direction(self) -> Vec3:
        return self.dir

    def at(self, t: float) -> Point3:
        return self.orig + self.dir * t


class RayList:
    def __init__(self, origin: np.ndarray, direction: np.ndarray) -> None:
        # shape: n * 3
        self.orig = origin
        self.dir = direction

    def origin(self) -> np.ndarray:
        return self.orig

    def direction(self) -> np.ndarray:
        return self.dir

    def at(self, t: np.ndarray) -> np.ndarray:
        # t's shape: n * 1
        return self.orig + np.transpose(np.transpose(self.dir) * t)
