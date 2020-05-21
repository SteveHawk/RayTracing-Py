from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np  # type: ignore
from utils.ray import Ray
from utils.vec3 import Vec3, Color
from utils.hittable import HitRecord
from utils.rtweekend import random_float


class Material(ABC):
    @abstractmethod
    def scatter(self, r_in: Ray, rec: HitRecord) \
            -> Optional[Tuple[Ray, Color]]:
        return NotImplemented


class Lambertian(Material):
    def __init__(self, a: Color) -> None:
        self.albedo = a

    def scatter(self, r_in: Ray, rec: HitRecord) \
            -> Optional[Tuple[Ray, Color]]:
        scatter_direction: Vec3 = rec.normal + Vec3.random_unit_vector()
        scattered = Ray(rec.p, scatter_direction)
        attenuation = self.albedo
        return scattered, attenuation


class Hemisphere(Material):
    def __init__(self, a: Color) -> None:
        self.albedo = a

    def scatter(self, r_in: Ray, rec: HitRecord) \
            -> Optional[Tuple[Ray, Color]]:
        scatter_direction: Vec3 = Vec3.random_in_hemisphere(rec.normal)
        scattered = Ray(rec.p, scatter_direction)
        attenuation = self.albedo
        return scattered, attenuation


class Metal(Material):
    def __init__(self, a: Color, f: float) -> None:
        self.albedo = a
        self.fuzz = f if f < 1 else 1

    def scatter(self, r_in: Ray, rec: HitRecord) \
            -> Optional[Tuple[Ray, Color]]:
        reflected: Vec3 = r_in.direction().unit_vector().reflect(rec.normal) \
            + Vec3.random_in_unit_sphere() * self.fuzz
        scattered = Ray(rec.p, reflected)
        attenuation = self.albedo
        if scattered.direction() @ rec.normal > 0:
            return scattered, attenuation
        return None


class Dielectric(Material):
    def __init__(self, ri: float) -> None:
        self.ref_idx = ri  # refractive indices

    def scatter(self, r_in: Ray, rec: HitRecord) \
            -> Optional[Tuple[Ray, Color]]:
        if rec.front_face:
            etai_over_etat = 1 / self.ref_idx
        else:
            etai_over_etat = self.ref_idx

        unit_direction: Vec3 = r_in.direction().unit_vector()
        cos_theta: float = min(-unit_direction @ rec.normal, 1)
        sin_theta: float = np.sqrt(1 - cos_theta**2)
        reflect_prob: float = self.schlick(cos_theta, etai_over_etat)

        if etai_over_etat * sin_theta > 1 or random_float() < reflect_prob:
            # total internal reflection
            reflected: Vec3 = unit_direction.reflect(rec.normal)
            scattered = Ray(rec.p, reflected)
        else:
            # refraction
            refracted: Vec3 = unit_direction.refract(
                rec.normal, etai_over_etat
            )
            scattered = Ray(rec.p, refracted)

        attenuation = Color(1, 1, 1)
        return scattered, attenuation

    @staticmethod
    def schlick(cosine: float, ref_idx: float) -> float:
        r0 = (1 - ref_idx) / (1 + ref_idx)
        r0 **= 2
        return r0 + (1 - r0) * ((1 - cosine) ** 5)
