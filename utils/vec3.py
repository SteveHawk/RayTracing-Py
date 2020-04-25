from __future__ import annotations
import numpy as np  # type: ignore
from typing import Union


class Vec3:
    def __init__(self, e0: float = 0, e1: float = 0, e2: float = 0) -> None:
        self.e: np.ndarray = np.array([e0, e1, e2], dtype=np.double)

    def x(self) -> float:
        return self.e[0]

    def y(self) -> float:
        return self.e[1]

    def z(self) -> float:
        return self.e[2]

    def __getitem__(self, idx: int) -> float:
        return self.e[idx]

    def __str__(self) -> str:
        return f"{self.e[0]} {self.e[1]} {self.e[2]}"

    def length_squared(self) -> float:
        return np.sum(self.e ** 2)

    def length(self) -> float:
        return np.sqrt(self.length_squared())

    def __add__(self, v: Vec3) -> Vec3:
        return Vec3(*(self.e + v.e))

    def __neg__(self) -> Vec3:
        return Vec3(*(-self.e))

    def __sub__(self, v: Vec3) -> Vec3:
        return self + (-v)

    def __mul__(self, v: Union[Vec3, int, float]) -> Vec3:
        if isinstance(v, Vec3):
            return Vec3(*(self.e * v.e))
        elif isinstance(v, (int, float)):
            return Vec3(*(self.e * v))
        raise NotImplementedError

    def __matmul__(self, v: Vec3) -> float:
        return self.e @ v.e

    def __truediv__(self, t: float) -> Vec3:
        return self * (1/t)

    def __iadd__(self, v: Vec3) -> Vec3:
        self.e += v
        return self

    def __imul__(self, v: Union[Vec3, int, float]) -> Vec3:
        if isinstance(v, Vec3):
            self.e *= v.e
        elif isinstance(v, (int, float)):
            self.e *= v
        else:
            return NotImplementedError
        return self

    def __itruediv__(self, t: float) -> Vec3:
        self *= (1/t)
        return self

    def cross(self, v: Vec3) -> Vec3:
        return Vec3(*np.cross(self.e, v.e))

    def unit_vector(self) -> Vec3:
        return (self / self.length())
