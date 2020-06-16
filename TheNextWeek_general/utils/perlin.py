import numpy as np  # type: ignore
from typing import List
from utils.vec3 import Point3
from utils.rtweekend import random_int, random_float_list


class Perlin:
    def __init__(self) -> None:
        self.point_count = 256
        self.ranfloat: List[float] = random_float_list(self.point_count)
        self.perm_x: List[int] = self.perlin_generate_perm()
        self.perm_y: List[int] = self.perlin_generate_perm()
        self.perm_z: List[int] = self.perlin_generate_perm()

    def perlin_generate_perm(self) -> List[int]:
        p = np.arange(self.point_count, dtype=np.int32)
        for i in range(self.point_count - 1, 0, -1):
            target = random_int(0, i)
            p[i], p[target] = p[target], p[i]
        return p

    def noise(self, p: Point3) -> float:
        u = p.x() - np.floor(p.x())
        v = p.y() - np.floor(p.y())
        w = p.z() - np.floor(p.z())

        i = int(4 * p.x()) & 255
        j = int(4 * p.y()) & 255
        k = int(4 * p.z()) & 255

        return self.ranfloat[self.perm_x[i] ^ self.perm_y[j] ^ self.perm_z[k]]
