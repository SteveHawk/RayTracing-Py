from typing import Optional
from utils.hittable import Hittable, HitRecord, FlipFace
from utils.hittable_list import HittableList
from utils.vec3 import Point3
from utils.material import Material
from utils.aarect import XYRect, XZRect, YZRect
from utils.ray import Ray
from utils.aabb import AABB


class Box(Hittable):
    def __init__(self, p0: Point3, p1: Point3, mat: Material) -> None:
        self.box_min = p0
        self.box_max = p1

        self.sides = HittableList()
        self.sides.add(XYRect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), mat))
        self.sides.add(
            FlipFace(XYRect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), mat))
        )
        self.sides.add(XZRect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), mat))
        self.sides.add(
            FlipFace(XZRect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), mat))
        )
        self.sides.add(YZRect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), mat))
        self.sides.add(
            FlipFace(YZRect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), mat))
        )

    def hit(self, r: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        return self.sides.hit(r, t_min, t_max)

    def bounding_box(self, t0: float, t1: float) -> Optional[AABB]:
        return AABB(self.box_min, self.box_max)
