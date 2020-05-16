import numpy as np
from typing import Optional
from utils.vec3 import Vec3, Point3
from utils.ray import Ray
from utils.hittable import hittable, hit_record


class sphere(hittable):
    def __init__(self, center: Point3 = Point3(), r: float = 0):
        self.center = center
        self.radius = r

    def hit(self, r: Ray, t_min: float, t_max: float) -> Optional[hit_record]:
        oc: Vec3 = r.origin() - self.center
        a: float = r.direction().length_squared()
        half_b: float = oc @ r.direction()
        c: float = oc.length_squared() - self.radius**2
        discriminant: float = half_b**2 - a*c

        if discriminant > 0:
            root: float = np.sqrt(discriminant)
            t_0: float = (-half_b - root) / a
            t_1: float = (-half_b + root) / a

            if t_min < t_0 < t_max:
                t = t_0
            elif t_min < t_1 < t_max:
                t = t_1
            else:
                return None

            p: Point3 = r.at(t)
            normal: Vec3 = (p - self.center) / self.radius
            return hit_record(p, normal, t)

        return None
