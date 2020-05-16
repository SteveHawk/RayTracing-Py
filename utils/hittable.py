from abc import ABC, abstractmethod
from typing import Optional
from utils.vec3 import Vec3, Point3
from utils.ray import Ray


class hit_record:
    def __init__(self, point: Point3, normal: Vec3, t: float) -> None:
        self.p = point
        self.normal = normal
        self.t = t


class hittable(ABC):
    @abstractmethod
    def hit(self, r: Ray, t_min: float, t_max: float) -> Optional[hit_record]:
        return NotImplemented
