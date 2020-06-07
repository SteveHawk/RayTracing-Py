import numpy as np  # type: ignore
from typing import Optional
from utils.vec3 import Vec3, Point3
from utils.hittable import Hittable, HitRecord
from utils.material import Material
from utils.ray import Ray


class MovingSphere(Hittable):
    def __init__(self, cen0: Point3, cen1: Point3,
                 t0: float, t1: float, r: float, m: Material) -> None:
        self.center0 = cen0
        self.center1 = cen1
        self.time0 = t0
        self.time1 = t1
        self.radius = r
        self.material = m

    def hit(self, r: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        oc: Vec3 = r.origin() - self.center(r.time())
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

            point: Point3 = r.at(t)
            outward_normal: Vec3 = (
                (point - self.center(r.time())) / self.radius
            )

            rec = HitRecord(point, t, self.material)
            rec.set_face_normal(r, outward_normal)
            return rec

        return None

    def center(self, time: float) -> Point3:
        return (
            self.center0 + (
                ((time - self.time0) / (self.time1 - self.time0))
                * (self.center1 - self.center0)
            )
        )
