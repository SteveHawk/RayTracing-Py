from __future__ import annotations
import numpy as np  # type: ignore
from typing import Tuple
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

    def __len__(self) -> int:
        return len(self.orig)

    def __getitem__(self, idx: int) -> Ray:
        return Ray(Point3(*self.orig[idx]), Vec3(*self.dir[idx]))

    def __setitem__(self, idx: int, r: Ray) -> None:
        self.orig[idx] = r.orig.e
        self.dir[idx] = r.dir.e

    def at(self, t: np.ndarray) -> np.ndarray:
        # t's shape: n * 1
        return self.orig + np.transpose(np.transpose(self.dir) * t)

    @staticmethod
    def single(r: Ray) -> RayList:
        return RayList(np.array([r.orig.e]), np.array([r.dir.e]))
