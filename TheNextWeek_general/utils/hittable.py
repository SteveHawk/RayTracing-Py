from __future__ import annotations
import numpy as np  # type: ignore
from abc import ABC, abstractmethod
from typing import Optional
from utils.vec3 import Vec3, Point3
from utils.ray import Ray
from utils.aabb import AABB
from utils.rtweekend import degrees_to_radians

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


class RotateY(Hittable):
    def __init__(self, obj: Hittable, angle: float) -> None:
        radians = degrees_to_radians(angle)
        self.sin_theta = np.sin(radians)
        self.cos_theta = np.cos(radians)
        self.obj = obj
        self.bbox = obj.bounding_box(0, 1)

        if self.bbox is None:
            raise ValueError

        point_min = Point3(np.inf, np.inf, np.inf)
        point_max = Point3(-np.inf, -np.inf, -np.inf)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    x = i*self.bbox.max().x() + (1-i)*self.bbox.min().x()
                    y = j*self.bbox.max().y() + (1-j)*self.bbox.min().y()
                    z = k*self.bbox.max().z() + (1-k)*self.bbox.min().z()

                    newx = self.cos_theta * x + self.sin_theta * z
                    newz = -self.sin_theta * x + self.cos_theta * z

                    new = Vec3(newx, y, newz)
                    for c in range(3):
                        point_min[c] = np.minimum(point_min[c], new[c])
                        point_max[c] = np.maximum(point_max[c], new[c])

        self.bbox = AABB(point_min, point_max)

    def hit(self, r: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        origin = r.origin().copy()
        direction = r.direction().copy()

        origin[0] = self.cos_theta*r.origin()[0] - self.sin_theta*r.origin()[2]
        origin[2] = self.sin_theta*r.origin()[0] + self.cos_theta*r.origin()[2]
        direction[0] = \
            self.cos_theta*r.direction()[0] - self.sin_theta*r.direction()[2]
        direction[2] = \
            self.sin_theta*r.direction()[0] + self.cos_theta*r.direction()[2]

        rotated_r = Ray(origin, direction, r.time())
        rec = self.obj.hit(rotated_r, t_min, t_max)
        if rec is None:
            return None

        p = rec.p.copy()
        normal = rec.normal.copy()

        p[0] = self.cos_theta*rec.p[0] + self.sin_theta*rec.p[2]
        p[2] = -self.sin_theta*rec.p[0] + self.cos_theta*rec.p[2]
        normal[0] = \
            self.cos_theta*rec.normal[0] + self.sin_theta*rec.normal[2]
        normal[2] = \
            -self.sin_theta*rec.normal[0] + self.cos_theta*rec.normal[2]

        rec.p = p
        rec.set_face_normal(rotated_r, normal)

        return rec

    def bounding_box(self, t0: float, t1: float) -> Optional[AABB]:
        return self.bbox
