from __future__ import annotations
import numpy as np  # type: ignore
from PIL import Image  # type: ignore
from utils.vec3 import Color


class Img:
    def __init__(self, w: int, h: int) -> None:
        self.frame: np.ndarray = np.zeros((h, w, 3), dtype=np.double)

    def set_array(self, array: np.ndarray) -> None:
        self.frame = array

    def write_pixel(self, w: int, h: int, v: Color) -> None:
        self.frame[h][w] = v.e

    def save(self, path: str, show: bool = False) -> None:
        im = Image.fromarray(np.uint8(self.frame * 255))
        im.save(path)
        if show:
            im.show()
