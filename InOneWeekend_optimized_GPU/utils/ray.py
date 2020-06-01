from __future__ import annotations
import cupy as cp  # type: ignore
from typing import Tuple
from utils.vec3 import Vec3, Point3, Vec3List


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
    def __init__(self, origin: Vec3List, direction: Vec3List) -> None:
        # shape: n * 3
        self.orig = origin
        self.dir = direction

    def origin(self) -> Vec3List:
        return self.orig

    def direction(self) -> Vec3List:
        return self.dir

    def __len__(self) -> int:
        return len(self.orig)

    def __getitem__(self, idx: int) -> Ray:
        return Ray(self.orig[idx], self.dir[idx])

    def __setitem__(self, idx: int, r: Ray) -> None:
        self.orig[idx] = r.orig
        self.dir[idx] = r.dir

    def __add__(self, r: RayList) -> RayList:
        return RayList(
            self.orig + r.orig,
            self.dir + r.dir
        )

    def at(self, t: cp.ndarray) -> Vec3List:
        # t's shape: n * 1
        return self.orig + self.dir.mul_ndarray(t)

    @staticmethod
    def single(r: Ray) -> RayList:
        return RayList(cp.array([r.orig.e]), cp.array([r.dir.e]))

    @staticmethod
    def new_empty(length: int) -> RayList:
        return RayList(Vec3List.new_empty(length),
                       Vec3List.new_empty(length))

    @staticmethod
    def new_zero(length: int) -> RayList:
        return RayList(Vec3List.new_zero(length),
                       Vec3List.new_zero(length))
