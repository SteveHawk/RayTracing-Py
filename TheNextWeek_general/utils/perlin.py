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
        i = np.floor(p.x())
        j = np.floor(p.y())
        k = np.floor(p.z())
        c = np.empty((2, 2, 2), dtype=np.float32)

        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    c[di][dj][dk] = self.ranfloat[
                        self.perm_x[int(i+di) & 255]
                        ^ self.perm_y[int(j+dj) & 255]
                        ^ self.perm_z[int(k+dk) & 255]
                    ]

        return Perlin.trilinear_interp(c, u, v, w)

    @staticmethod
    def trilinear_interp(c: List[List[List[float]]],
                         u: float, v: float, w: float) -> float:
        accum: float = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    accum += ((i*u + (1-i) * (1-u))
                              * (j*v + (1-j) * (1-v))
                              * (k*w + (1-k) * (1-w))
                              ) * c[i][j][k]
        return accum
