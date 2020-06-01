from abc import ABC, abstractmethod
from typing import Tuple, Optional
import cupy as cp  # type: ignore
from utils.ray import RayList
from utils.vec3 import Vec3, Color, Vec3List
from utils.hittable import HitRecordList
from utils.rtweekend import random_float_list


class Material(ABC):
    @abstractmethod
    def __init__(self, idx: int) -> None:
        self.idx = idx

    @abstractmethod
    def scatter(self, r_in: RayList, rec: HitRecordList) \
            -> Tuple[RayList, Vec3List]:
        return NotImplemented


class Lambertian(Material):
    def __init__(self, a: Color, idx: int) -> None:
        self.albedo = a
        self.idx = idx

    def scatter(self, r_in: RayList, rec: HitRecordList) \
            -> Tuple[RayList, Vec3List]:
        condition = (rec.t > 0) & rec.front_face

        scatter_direction = rec.normal + Vec3.random_unit_vector(len(r_in))
        scattered = RayList(
            rec.p.mul_ndarray(condition),
            scatter_direction.mul_ndarray(condition)
        )
        attenuation = Vec3List.from_array(condition) * self.albedo

        return scattered, attenuation


class Hemisphere(Material):
    def __init__(self, a: Color, idx: int) -> None:
        self.albedo = a
        self.idx = idx

    def scatter(self, r_in: RayList, rec: HitRecordList) \
            -> Tuple[RayList, Vec3List]:
        condition = (rec.t > 0) & rec.front_face

        scatter_direction = Vec3.random_in_hemisphere(rec.normal)
        scattered = RayList(
            rec.p.mul_ndarray(condition),
            scatter_direction.mul_ndarray(condition)
        )
        attenuation = Vec3List.from_array(condition) * self.albedo

        return scattered, attenuation


class Metal(Material):
    def __init__(self, a: Color, f: float, idx: int) -> None:
        self.albedo = a
        self.fuzz = f if f < 1 else 1
        self.idx = idx

    def scatter(self, r_in: RayList, rec: HitRecordList) \
            -> Tuple[RayList, Vec3List]:
        condition = (rec.t > 0) & rec.front_face

        reflected = (
            r_in.direction().unit_vector().reflect(rec.normal)
            + Vec3.random_in_unit_sphere_list(len(r_in)) * self.fuzz
        )

        condition = condition & (reflected @ rec.normal > 0)
        scattered = RayList(
            rec.p.mul_ndarray(condition),
            reflected.mul_ndarray(condition)
        )
        attenuation = Vec3List.from_array(condition) * self.albedo

        return scattered, attenuation


class Dielectric(Material):
    def __init__(self, ri: float, idx: int) -> None:
        self.ref_idx = ri  # refractive indices
        self.idx = idx

    def scatter(self, r_in: RayList, rec: HitRecordList) \
            -> Tuple[RayList, Vec3List]:
        etai_over_etat = cp.where(
            rec.front_face, 1 / self.ref_idx, self.ref_idx
        )

        unit_direction = r_in.direction().unit_vector()
        cos_theta = -unit_direction @ rec.normal
        cos_theta = cp.where(cos_theta > 1, 1, cos_theta)
        sin_theta = cp.sqrt(1 - cos_theta**2)
        reflect_prob = self.schlick(cos_theta, etai_over_etat)

        reflect_condition = (
            (etai_over_etat * sin_theta > 1)
            | (random_float_list(len(r_in)) < reflect_prob)
        )
        # total internal reflection
        reflected = (unit_direction.mul_ndarray(reflect_condition)).reflect(
            rec.normal.mul_ndarray(reflect_condition)
        )
        # refraction
        refracted = (unit_direction.mul_ndarray(~reflect_condition)).refract(
            rec.normal.mul_ndarray(~reflect_condition), etai_over_etat
        )
        direction = reflected + refracted

        condition = rec.t > 0
        scattered = RayList(
            rec.p.mul_ndarray(condition),
            direction.mul_ndarray(condition)
        )
        attenuation = Vec3List.from_array(condition) * Color(1, 1, 1)
        return scattered, attenuation

    @staticmethod
    def schlick(cosine: cp.ndarray, ref_idx: cp.ndarray) -> cp.ndarray:
        r0 = (1 - ref_idx) / (1 + ref_idx)
        r0 **= 2
        return r0 + (1 - r0) * ((1 - cosine) ** 5)
