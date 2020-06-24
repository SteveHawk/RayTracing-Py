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
        self.u: float
        self.v: float

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


class FlipFace(Hittable):
    def __init__(self, obj: Hittable):
        self.obj = obj

    def hit(self, r: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        rec = self.obj.hit(r, t_min, t_max)
        if rec is None:
            return None
        rec.front_face = not rec.front_face
        return rec

    def bounding_box(self, t0: float, t1: float) -> Optional[AABB]:
        return self.obj.bounding_box(t0, t1)


class Translate(Hittable):
    def __init__(self, obj: Hittable, displacement: Vec3) -> None:
        self.obj = obj
        self.offset = displacement

    def hit(self, r: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        # remove the offset to hit the real object
        moved_r = Ray(r.origin() - self.offset, r.direction(), r.time())
        rec = self.obj.hit(moved_r, t_min, t_max)
        if rec is None:
            return None

        # add the offset back to simulate the move
        rec.p += self.offset
        rec.set_face_normal(moved_r, rec.normal)
        return rec

    def bounding_box(self, t0: float, t1: float) -> Optional[AABB]:
        box = self.obj.bounding_box(t0, t1)
        if box is None:
            return None

        output_box = AABB(
            box.min() + self.offset,
            box.max() + self.offset
        )
        return output_box
