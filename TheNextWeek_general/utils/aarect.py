from typing import Optional
from utils.hittable import Hittable, HitRecord
from utils.vec3 import Vec3, Point3
from utils.ray import Ray
from utils.aabb import AABB
from utils.material import Material


class XYRect(Hittable):
    def __init__(self, x0: float, x1: float, y0: float,
                 y1: float, k: float, mat: Material) -> None:
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.k = k
        self.material = mat

    def hit(self, r: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        t = (self.k - r.origin().z()) / r.direction().z()
        if t < t_min or t > t_max:
            return None
        x = r.origin().x() + t*r.direction().x()
        y = r.origin().y() + t*r.direction().y()
        if (x < self.x0) or (x > self.x1) or (y < self.y0) or (y > self.y1):
            return None

        rec = HitRecord(r.at(t), t, self.material)
        rec.set_face_normal(r, Vec3(0, 0, 1))
        rec.u = (x - self.x0) / (self.x1 - self.x0)
        rec.v = (y - self.y0) / (self.y1 - self.y0)
        return rec

    def bounding_box(self, t0: float, t1: float) -> Optional[AABB]:
        output_box = AABB(
            Point3(self.x0, self.y0, self.k-0.0001),
            Point3(self.x1, self.y1, self.k+0.0001)
        )
        return output_box


class XZRect(Hittable):
    def __init__(self, x0: float, x1: float, z0: float,
                 z1: float, k: float, mat: Material) -> None:
        self.x0 = x0
        self.x1 = x1
        self.z0 = z0
        self.z1 = z1
        self.k = k
        self.material = mat

    def hit(self, r: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        t = (self.k - r.origin().y()) / r.direction().y()
        if t < t_min or t > t_max:
            return None
        x = r.origin().x() + t*r.direction().x()
        z = r.origin().z() + t*r.direction().z()
        if (x < self.x0) or (x > self.x1) or (z < self.z0) or (z > self.z1):
            return None

        rec = HitRecord(r.at(t), t, self.material)
        rec.set_face_normal(r, Vec3(0, 1, 0))
        rec.u = (x - self.x0) / (self.x1 - self.x0)
        rec.v = (z - self.z0) / (self.z1 - self.z0)
        return rec

    def bounding_box(self, t0: float, t1: float) -> Optional[AABB]:
        output_box = AABB(
            Point3(self.x0, self.k-0.0001, self.z0),
            Point3(self.x1, self.k+0.0001, self.z1)
        )
        return output_box


class YZRect(Hittable):
    def __init__(self, y0: float, y1: float, z0: float,
                 z1: float, k: float, mat: Material) -> None:
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1
        self.k = k
        self.material = mat

    def hit(self, r: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        t = (self.k - r.origin().x()) / r.direction().x()
        if t < t_min or t > t_max:
            return None
        y = r.origin().y() + t*r.direction().y()
        z = r.origin().z() + t*r.direction().z()
        if (y < self.y0) or (y > self.y1) or (z < self.z0) or (z > self.z1):
            return None

        rec = HitRecord(r.at(t), t, self.material)
        rec.set_face_normal(r, Vec3(1, 0, 0))
        rec.u = (y - self.y0) / (self.y1 - self.y0)
        rec.v = (z - self.z0) / (self.z1 - self.z0)
        return rec

    def bounding_box(self, t0: float, t1: float) -> Optional[AABB]:
        output_box = AABB(
            Point3(self.k-0.0001, self.y0, self.z0),
            Point3(self.k+0.0001, self.y1, self.z1)
        )
        return output_box
