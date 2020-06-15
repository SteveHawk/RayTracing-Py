from abc import ABC, abstractmethod
from typing import Union
from utils.vec3 import Vec3, Color, Point3


class Texture(ABC):
    @abstractmethod
    def value(self, u: float, v: float, p: Point3) -> Color:
        return NotImplemented


class SolidColor(Texture):
    def __init__(self, r: Union[float, Color],
                 g: float = None, b: float = None) -> None:
        if isinstance(r, Color):
            self.color_value = r
        elif g is not None and b is not None:
            self.color_value = Color(r, g, b)
        else:
            raise ValueError

    def value(self, u: float, v: float, p: Point3) -> Color:
        return self.color_value
