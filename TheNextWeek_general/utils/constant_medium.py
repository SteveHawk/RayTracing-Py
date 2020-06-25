import numpy as np  # type: ignore
from typing import Optional
from utils.hittable import Hittable, HitRecord
from utils.texture import Texture
from utils.material import Material, Isotropic
from utils.ray import Ray
from utils.aabb import AABB
from utils.vec3 import Vec3
from utils.rtweekend import random_float


class ConstantMedium(Hittable):
    def __init__(self, b: Hittable, d: float, a: Texture):
        self.boundary = b
        self.neg_inv_density: float = -1 / d
        self.phase_function: Material = Isotropic(a)

    def hit(self, r: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        rec1 = self.boundary.hit(r, -np.inf, np.inf)
        if rec1 is None:
            return None
        rec2 = self.boundary.hit(r, rec1.t+0.0001, np.inf)
        if rec2 is None:
            return None

        if rec1.t < t_min:
            rec1.t = t_min
        if rec1.t < 0:
            rec1.t = 0
        if rec2.t > t_max:
            rec2.t = t_max

        if rec1.t >= rec2.t:
            return None

        ray_length = r.direction().length()
        distance_inside_boundary = (rec2.t - rec1.t) * ray_length
        hit_distance = self.neg_inv_density * np.log(random_float())

        if hit_distance > distance_inside_boundary:
            return None

        t = rec1.t + hit_distance / ray_length
        p = r.at(t)
        rec = HitRecord(p, t, self.phase_function)
        rec.normal = Vec3(1, 0, 0)
        rec.front_face = True

        return rec

    def bounding_box(self, t0: float, t1: float) -> Optional[AABB]:
        return self.boundary.bounding_box(t0, t1)
