from abc import ABC, abstractmethod
from utils.vec3 import Vec3, Color, Point3


class Texture(ABC):
    @abstractmethod
    def value(self, u: float, v: float, p: Point3) -> Color:
        return NotImplemented


class SolidColor(Texture):
    def __init__(self, c: Color) -> None:
        self.color_value = c

    def value(self, u: float, v: float, p: Point3) -> Color:
        return self.color_value
