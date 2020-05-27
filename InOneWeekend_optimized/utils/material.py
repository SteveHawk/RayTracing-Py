from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np  # type: ignore
from utils.ray import RayList
from utils.vec3 import Vec3, Color
from utils.hittable import HitRecordList
from utils.rtweekend import random_float_list


class Material(ABC):
    @abstractmethod
    def __init__(self, idx: int) -> None:
        self.idx = idx

    @abstractmethod
    def scatter(self, r_in: RayList, rec: HitRecordList) \
            -> Tuple[RayList, np.ndarray]:
        return NotImplemented


class Lambertian(Material):
    def __init__(self, a: Color, idx: int) -> None:
        self.albedo = a
        self.idx = idx

    def scatter(self, r_in: RayList, rec: HitRecordList) \
            -> Tuple[RayList, np.ndarray]:
        condition = (rec.t > 0) & rec.front_face

        scatter_direction = rec.normal + Vec3.random_unit_vector(len(r_in))
        scattered = RayList(
            np.transpose(np.transpose(rec.p) * condition),
            np.transpose(np.transpose(scatter_direction) * condition)
        )
        attenuation = self.albedo.e * np.transpose(np.tile(condition, (3, 1)))

        return scattered, attenuation


class Hemisphere(Material):
    def __init__(self, a: Color, idx: int) -> None:
        self.albedo = a
        self.idx = idx

    def scatter(self, r_in: RayList, rec: HitRecordList) \
            -> Tuple[RayList, np.ndarray]:
        condition = (rec.t > 0) & rec.front_face

        scatter_direction = Vec3.random_in_hemisphere(rec.normal)
        scattered = RayList(
            np.transpose(np.transpose(rec.p) * condition),
            np.transpose(np.transpose(scatter_direction) * condition)
        )
        attenuation = self.albedo.e * np.transpose(np.tile(condition, (3, 1)))

        return scattered, attenuation


class Metal(Material):
    def __init__(self, a: Color, f: float, idx: int) -> None:
        self.albedo = a
        self.fuzz = f if f < 1 else 1
        self.idx = idx

    def scatter(self, r_in: RayList, rec: HitRecordList) \
            -> Tuple[RayList, np.ndarray]:
        condition = (rec.t > 0) & rec.front_face

        r_in_dir = r_in.direction()
        unit_dir = np.transpose(
            np.transpose(r_in_dir) / np.sqrt((r_in_dir ** 2).sum(axis=1))
        )
        reflect_ray = (
            unit_dir - np.transpose(
                np.transpose(rec.normal) * (unit_dir * rec.normal).sum(axis=1)
            ) * 2
        )
        reflected = (
            reflect_ray
            + Vec3.random_in_unit_sphere_list(len(r_in)) * self.fuzz
        )

        condition = condition & ((reflected * rec.normal).sum(axis=1) > 0)
        scattered = RayList(
            np.transpose(np.transpose(rec.p) * condition),
            np.transpose(np.transpose(reflected) * condition)
        )
        attenuation = self.albedo.e * np.transpose(np.tile(condition, (3, 1)))

        return scattered, attenuation


class Dielectric(Material):
    def __init__(self, ri: float, idx: int) -> None:
        self.ref_idx = ri  # refractive indices
        self.idx = idx

    def scatter(self, r_in: RayList, rec: HitRecordList) \
            -> Tuple[RayList, np.ndarray]:
        etai_over_etat = np.where(
            rec.front_face, 1 / self.ref_idx, self.ref_idx
        )

        r_in_dir = r_in.direction()
        unit_direction = np.transpose(
            np.transpose(r_in_dir) / np.sqrt((r_in_dir ** 2).sum(axis=1))
        )
        cos_theta = (-unit_direction * rec.normal).sum(axis=1)
        cos_theta = np.where(cos_theta > 1, 1, cos_theta)
        sin_theta = np.sqrt(1 - cos_theta**2)
        reflect_prob: float = self.schlick(cos_theta, etai_over_etat)

        condition = np.transpose(np.tile(
            (etai_over_etat * sin_theta > 1)
            | (random_float_list(len(r_in)) < reflect_prob),
            (3, 1)
        ))
        # total internal reflection
        reflected = (
            unit_direction - np.transpose(
                np.transpose(rec.normal)
                * (unit_direction * rec.normal).sum(axis=1)
            ) * 2
        )
        # refraction
        r_out_parallel = np.transpose(np.transpose(
            unit_direction
            + np.transpose(np.transpose(rec.normal) * cos_theta)
        ) * etai_over_etat)
        r_out_parallel_length2 = (r_out_parallel ** 2).sum(axis=1)
        r_out_prep = np.transpose(
            np.transpose(rec.normal) * (-np.sqrt(1 - r_out_parallel_length2))
        )
        refracted = r_out_parallel + r_out_prep

        direction = np.where(condition, reflected, refracted)

        condition_0 = rec.t > 0
        scattered = RayList(
            np.transpose(np.transpose(rec.p) * condition_0),
            np.transpose(np.transpose(direction) * condition_0)
        )
        attenuation = (
            Color(1, 1, 1).e * np.transpose(np.tile(condition_0, (3, 1)))
        )
        return scattered, attenuation

    @staticmethod
    def schlick(cosine: np.ndarray, ref_idx: np.ndarray) -> np.ndarray:
        r0 = (1 - ref_idx) / (1 + ref_idx)
        r0 **= 2
        return r0 + (1 - r0) * ((1 - cosine) ** 5)
