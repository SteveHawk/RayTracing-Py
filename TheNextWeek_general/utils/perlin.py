import numpy as np  # type: ignore
from typing import List
from utils.vec3 import Vec3, Point3
from utils.rtweekend import random_int, random_float_list


class Perlin:
    def __init__(self) -> None:
        self.point_count = 256

        self.ranvec: List[Vec3] = list()
        for i in range(self.point_count):
            self.ranvec.append(Vec3.random(-1, 1).unit_vector())

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
        c: List[List[List[Vec3]]] = np.empty((2, 2, 2), dtype=object)

        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    c[di][dj][dk] = self.ranvec[
                        self.perm_x[int(i+di) & 255]
                        ^ self.perm_y[int(j+dj) & 255]
                        ^ self.perm_z[int(k+dk) & 255]
                    ]

        return Perlin.trilinear_interp(c, u, v, w)

    def turb(self, p: Point3, depth: int = 7) -> float:
        accum: float = 0
        weight: float = 1
        temp_p = p

        for i in range(depth):
            accum += weight * self.noise(temp_p)
            weight *= 0.5
            temp_p *= 2

        return np.abs(accum)

    @staticmethod
    def trilinear_interp(c: List[List[List[Vec3]]],
                         u: float, v: float, w: float) -> float:
        u = u*u * (3 - 2*u)
        v = v*v * (3 - 2*v)
        w = w*w * (3 - 2*w)
        accum: float = 0

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    weight_v = Vec3(u-i, v-j, w-k)
                    accum += (
                        (i*u + (1-i) * (1-u))
                        * (j*v + (1-j) * (1-v))
                        * (k*w + (1-k) * (1-w))
                        * (c[i][j][k] @ weight_v)
                    )
        return accum
