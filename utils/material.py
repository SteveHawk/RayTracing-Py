from abc import ABC, abstractmethod
from typing import Tuple, Optional
from utils.ray import Ray
from utils.vec3 import Vec3, Color
from utils.hittable import HitRecord


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
