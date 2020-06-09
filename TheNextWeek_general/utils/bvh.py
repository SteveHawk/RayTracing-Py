from typing import Optional
from utils.hittable import Hittable, HitRecord
from utils.ray import Ray
from utils.aabb import AABB
from utils.hittable_list import HittableList


class BVHNode(Hittable):
    def __init__(self, hlist: HittableList, time0: float, time1: float):
        self.left: Hittable
        self.right: Hittable
        self.box: AABB

    def hit(self, r: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        if not self.box.hit(r, t_min, t_max):
            return None

        rec_l = self.left.hit(r, t_min, t_max)
        rec_r = self.right.hit(r, t_min, t_max if rec_l is None else rec_l.t)

        if rec_r is not None:
            return rec_r
        return rec_l

    def bounding_box(self, t0: float, t1: float) -> Optional[AABB]:
        return self.box
