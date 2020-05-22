import numpy as np  # type: ignore
from typing import Optional, Union, List
from utils.vec3 import Vec3, Point3
from utils.ray import RayList
from utils.hittable import Hittable, HitRecord
from utils.material import Material


class Sphere(Hittable):
    def __init__(self, center: Point3, r: float, mat: Material) -> None:
        self.center = center
        self.radius = r
        self.material = mat

    def hit(self, r: RayList, t_min: float, t_max: Union[float, np.ndarray]) \
            -> List[Optional[HitRecord]]:
        if isinstance(t_max, (int, float, np.floating)):
            t_max_list = np.full(len(r), t_max)
        else:
            t_max_list = t_max

        oc: np.ndarray = r.origin() - self.center.e
        a: np.ndarray = (r.direction()**2).sum(axis=1)
        half_b: np.ndarray = (oc * r.direction()).sum(axis=1)
        c: np.ndarray = (oc**2).sum(axis=1) - self.radius**2
        discriminant_list: np.ndarray = half_b**2 - a*c

        positive_discriminant_list = np.clip(discriminant_list, 0, np.inf)
        root = np.sqrt(positive_discriminant_list)
        t_0 = (-half_b - root) / a
        t_1 = (-half_b + root) / a

        result: List[Optional[HitRecord]] = list()
        for i, discriminant in enumerate(discriminant_list):
            if discriminant > 0:
                if t_min < t_0[i] < t_max_list[i]:
                    t = t_0[i]
                elif t_min < t_1[i] < t_max_list[i]:
                    t = t_1[i]
                else:
                    result.append(None)
                    continue

                point: Point3 = r[i].at(t)
                outward_normal: Vec3 = (point - self.center) / self.radius

                rec = HitRecord(point, t, self.material)
                rec.set_face_normal(r[i], outward_normal)
                result.append(rec)
                continue

            result.append(None)
        return result
