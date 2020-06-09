from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from utils.vec3 import Vec3, Point3
from utils.ray import Ray
from utils.aabb import AABB

import typing
if typing.TYPE_CHECKING:
    from utils.material import Material


class HitRecord:
    def __init__(self, point: Point3, t: float, mat: Material) -> None:
        self.p = point
        self.t = t
        self.material = mat
        self.normal: Vec3
        self.front_face: bool

    def set_face_normal(self, r: Ray, outward_normal: Vec3) -> HitRecord:
        self.front_face = (r.direction() @ outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal
        return self


class Hittable(ABC):
    @abstractmethod
    def hit(self, r: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        return NotImplemented

    @abstractmethod
    def bounding_box(self, t0: float, t1: float) -> Optional[AABB]:
        return NotImplemented
