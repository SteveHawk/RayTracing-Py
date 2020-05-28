import numpy as np  # type: ignore
from typing import Optional, Union, List
from utils.vec3 import Vec3, Point3, Vec3List
from utils.ray import RayList
from utils.hittable import Hittable, HitRecordList
from utils.material import Material


class Sphere(Hittable):
    def __init__(self, center: Point3, r: float, mat: Material) -> None:
        self.center = center
        self.radius = r
        self.material = mat

    def hit(self, r: RayList, t_min: float, t_max: Union[float, np.ndarray]) \
            -> HitRecordList:
        if isinstance(t_max, (int, float, np.floating)):
            t_max_list = np.full(len(r), t_max)
        else:
            t_max_list = t_max

        oc: Vec3List = r.origin() - self.center
        a: np.ndarray = r.direction().length_squared()
        half_b: np.ndarray = oc @ r.direction()
        c: np.ndarray = oc.length_squared() - self.radius**2
        discriminant_list: np.ndarray = half_b**2 - a*c

        discriminant_condition = discriminant_list > 0
        if not discriminant_condition.any():
            return HitRecordList.new(len(r))

        positive_discriminant_list = (
            discriminant_list * discriminant_condition
        )
        root = np.sqrt(positive_discriminant_list)
        non_zero_a = a - (a == 0)
        t_0 = (-half_b - root) / non_zero_a
        t_1 = (-half_b + root) / non_zero_a

        t_0_condition = (
            (t_min < t_0) & (t_0 < t_max_list) & discriminant_condition
        )
        t_1_condition = (
            (t_min < t_1) & (t_1 < t_max_list)
            & (~t_0_condition) & discriminant_condition
        )
        t = np.where(t_0_condition, t_0, 0)
        t = np.where(t_1_condition, t_1, t)

        point = r.at(t)
        outward_normal = (point - self.center) / self.radius

        result = HitRecordList(
            point, t, np.full(len(r), self.material)
        ).set_face_normal(r, outward_normal)

        return result
