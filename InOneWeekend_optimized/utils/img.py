from __future__ import annotations
import numpy as np  # type: ignore
from PIL import Image  # type: ignore
from typing import List
from utils.vec3 import Color


class Img:
    def __init__(self, w: int, h: int) -> None:
        self.frame: np.ndarray = np.empty((h, w, 3), dtype=np.float32)

    def set_array(self, array: np.ndarray) -> None:
        self.frame = array

    def write_pixel(self, w: int, h: int, pixel_color: Color,
                    samples_per_pixel: int) -> None:
        color: Color = pixel_color / samples_per_pixel
        self.frame[h][w] = color.clamp(0, 0.999).gamma(2).e

    def write_pixel_list(self, h: int, pixel_color_list: np.ndarray,
                         samples_per_pixel: int) -> None:
        color = pixel_color_list / samples_per_pixel
        gamma: float = 2
        self.frame[h] = np.clip(color, 0, 0.999) ** (1 / gamma)

    def save(self, path: str, show: bool = False) -> None:
        im = Image.fromarray(np.uint8(self.frame * 255))
        im.save(path)
        if show:
            im.show()
