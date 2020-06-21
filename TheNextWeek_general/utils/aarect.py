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
