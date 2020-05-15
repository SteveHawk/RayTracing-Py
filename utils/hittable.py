from utils.vec3 import Vec3, Point3
from utils.ray import Ray
from abc import ABC, abstractmethod


class hit_record:
    def __init__(self, point: Point3, normal: Vec3, t: float) -> None:
        self.p = point
        self.normal = normal
        self.t = t


class hittable(ABC):
    @classmethod
    @abstractmethod
    def hit(cls, r: Ray, t_min: float, t_max: float, rec: hit_record) -> bool:
        return NotImplemented
