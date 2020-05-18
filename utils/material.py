from abc import ABC, abstractmethod
from typing import Tuple
from utils.ray import Ray
from utils.vec3 import Vec3, Color
from utils.hittable import HitRecord


class Material(ABC):
    @abstractmethod
    def scatter(self, r_in: Ray, rec: HitRecord) -> Tuple[Ray, Color]:
        return NotImplemented


class Lambertian(Material):
    def __init__(self, a: Color) -> None:
        self.albedo = a

    def scatter(self, r_in: Ray, rec: HitRecord) -> Tuple[Ray, Color]:
        scatter_direction: Vec3 = rec.normal + Vec3.random_unit_vector()
        scattered = Ray(rec.p, scatter_direction)
        attenuation = self.albedo
        return scattered, attenuation
