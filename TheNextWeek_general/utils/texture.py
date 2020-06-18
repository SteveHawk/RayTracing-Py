from abc import ABC, abstractmethod
import numpy as np  # type: ignore
from PIL import Image
from typing import Union
from utils.vec3 import Vec3, Color, Point3
from utils.perlin import Perlin


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


class CheckerTexture(Texture):
    def __init__(self, t0: Texture, t1: Texture) -> None:
        self.even = t0
        self.odd = t1

    def value(self, u: float, v: float, p: Point3) -> Color:
        sines = np.sin(10 * p.x()) * np.sin(10 * p.y()) * np.sin(10 * p.z())
        if sines < 0:
            return self.odd.value(u, v, p)
        else:
            return self.even.value(u, v, p)


class NoiseTexture(Texture):
    def __init__(self, scale: float = 1):
        self.noise = Perlin()
        self.scale = scale

    def value(self, u: float, v: float, p: Point3) -> Color:
        # return Color(1, 1, 1) * 0.5 * (1 + self.noise.noise(self.scale * p))
        # return Color(1, 1, 1) * self.noise.turb(self.scale * p)
        return (
            Color(1, 1, 1) * 0.5
            * (1 + np.sin(self.scale*p.z() + 10*self.noise.turb(p)))
        )


class ImageTexture(Texture):
    def __init__(self, filename: str) -> None:
        self.data = np.asarray(Image.open(filename))
        self.height, self.width, _ = self.data.shape

    def value(self, u: float, v: float, p: Vec3) -> Color:
        u = np.clip(u, 0, 1)
        v = 1 - np.clip(v, 0, 1)

        i = int(u * self.width)
        j = int(v * self.height)
        if i >= self.width:
            i = self.width - 1
        if j >= self.height:
            j = self.height - 1

        pixel = self.data[j][i] / 255
        return Color(*pixel)
