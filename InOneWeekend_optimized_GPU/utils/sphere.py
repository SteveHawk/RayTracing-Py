import cupy as cp  # type: ignore
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

    def hit(self, r: RayList, t_min: float, t_max: Union[float, cp.ndarray]) \
            -> HitRecordList:
        if isinstance(t_max, (int, float, cp.floating)):
            t_max_list = cp.full(len(r), t_max, cp.float32)
        else:
            t_max_list = t_max

        oc: Vec3List = r.origin() - self.center
        a: cp.ndarray = r.direction().length_squared()
        half_b: cp.ndarray = oc @ r.direction()
        c: cp.ndarray = oc.length_squared() - self.radius**2
        discriminant_list: cp.ndarray = half_b**2 - a*c

        discriminant_condition = discriminant_list > 0
        if not discriminant_condition.any():
            return HitRecordList.new(len(r)).set_compress_info(None)

        # Calculate t
        positive_discriminant_list = (
            discriminant_list * discriminant_condition
        )
        root = cp.sqrt(positive_discriminant_list)
        non_zero_a = a - (a == 0)
        t_0 = (-half_b - root) / non_zero_a
        t_1 = (-half_b + root) / non_zero_a

        # Choose t
        t_0_condition = (
            (t_min < t_0) & (t_0 < t_max_list) & discriminant_condition
        )
        t_1_condition = (
            (t_min < t_1) & (t_1 < t_max_list)
            & (~t_0_condition) & discriminant_condition
        )
        t = cp.where(t_0_condition, t_0, 0)
        t = cp.where(t_1_condition, t_1, t)

        # Compression
        condition = t > 0
        full_rate = condition.sum() / len(t)
        if full_rate > 0.5:
            idx = None
        else:
            idx = cp.where(condition)[0]
            t = t[idx]
            r = RayList(
                Vec3List(r.orig.get_ndarray(idx)),
                Vec3List(r.dir.get_ndarray(idx))
            )

        # Wrap up result
        point = r.at(t)
        outward_normal = (point - self.center) / self.radius

        result = HitRecordList(
            point, t, cp.full(len(r), self.material.idx, dtype=cp.int32)
        ).set_face_normal(r, outward_normal).set_compress_info(idx)

        return result
