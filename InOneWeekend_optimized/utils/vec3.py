from __future__ import annotations
import cupy as cp  # type: ignore
from typing import Union
from utils.rtweekend import random_float, random_float_list


class Vec3:
    def __init__(self, e0: float = 0, e1: float = 0, e2: float = 0) -> None:
        self.e: cp.ndarray = cp.array([e0, e1, e2], dtype=cp.float32)

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
        return self.e @ self.e

    def length(self) -> float:
        return cp.sqrt(self.length_squared())

    def __add__(self, v: Vec3) -> Vec3:
        return Vec3(*(self.e + v.e))

    def __neg__(self) -> Vec3:
        return Vec3(*(-self.e))

    def __sub__(self, v: Vec3) -> Vec3:
        return self + (-v)

    def __mul__(self, v: Union[Vec3, int, float]) -> Vec3:
        if isinstance(v, Vec3):
            return Vec3(*(self.e * v.e))
        # elif isinstance(v, (int, float, cp.floating)):
        return Vec3(*(self.e * v))
        # raise TypeError

    def __matmul__(self, v: Vec3) -> float:
        return self.e @ v.e

    def __truediv__(self, t: float) -> Vec3:
        return self * (1/t)

    def __iadd__(self, v: Vec3) -> Vec3:
        self.e += v.e
        return self

    def __imul__(self, v: Union[Vec3, int, float]) -> Vec3:
        if isinstance(v, Vec3):
            self.e *= v.e
        # elif isinstance(v, (int, float, cp.floating)):
        else:
            self.e *= v
            # raise TypeError
        return self

    def __itruediv__(self, t: float) -> Vec3:
        self *= (1/t)
        return self

    def cross(self, v: Vec3) -> Vec3:
        return Vec3(*cp.cross(self.e, v.e))

    def unit_vector(self) -> Vec3:
        length = self.length()
        if length == 0:
            return Vec3()
        return (self / length)

    def clamp(self, _min: float, _max: float) -> Vec3:
        return Vec3(*cp.clip(self.e, _min, _max))

    def gamma(self, gamma: float) -> Vec3:
        return Vec3(*(self.e ** (1 / gamma)))

    def reflect(self, n: Vec3) -> Vec3:
        return self - (n * (self @ n)) * 2

    def refract(self, normal: Vec3, etai_over_etat: float) -> Vec3:
        cos_theta: float = -self @ normal
        r_out_parallel: Vec3 = (self + normal*cos_theta) * etai_over_etat
        r_out_prep: Vec3 = \
            normal * (-cp.sqrt(1 - r_out_parallel.length_squared()))
        return r_out_parallel + r_out_prep

    @staticmethod
    def random(_min: float = 0, _max: float = 1) -> Vec3:
        return Vec3(*random_float_list(3, _min, _max))

    @staticmethod
    def random_in_unit_sphere() -> Vec3:
        """
        This method is modified from:
        https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/#better-choice-of-spherical-coordinates
        """
        u = random_float()
        v = random_float()
        theta = u * 2 * cp.pi
        phi = cp.arccos(2 * v - 1)
        r = cp.cbrt(random_float())
        sinTheta = cp.sin(theta)
        cosTheta = cp.cos(theta)
        sinPhi = cp.sin(phi)
        cosPhi = cp.cos(phi)
        x = r * sinPhi * cosTheta
        y = r * sinPhi * sinTheta
        z = r * cosPhi
        return Vec3(x, y, z)

    @staticmethod
    def random_in_unit_sphere_list(size: int) -> Vec3List:
        u = random_float_list(size)
        v = random_float_list(size)
        theta = u * 2 * cp.pi
        phi = cp.arccos(2 * v - 1)
        r = cp.cbrt(random_float_list(size))
        sinTheta = cp.sin(theta)
        cosTheta = cp.cos(theta)
        sinPhi = cp.sin(phi)
        cosPhi = cp.cos(phi)
        x = r * sinPhi * cosTheta
        y = r * sinPhi * sinTheta
        z = r * cosPhi
        return Vec3List(cp.transpose(cp.array([x, y, z])))

    @staticmethod
    def random_unit_vector(size: int) -> Vec3List:
        a = random_float_list(size, 0, 2 * cp.pi)
        z = random_float_list(size, -1, 1)
        r = cp.sqrt(1 - z**2)
        return Vec3List(cp.transpose(cp.array([r*cp.cos(a), r*cp.sin(a), z])))

    @staticmethod
    def random_in_hemisphere(normal: Vec3List) -> Vec3List:
        in_unit_sphere = Vec3.random_in_unit_sphere_list(len(normal))
        return Vec3List(cp.where(
            in_unit_sphere @ normal > 0, in_unit_sphere.e, -in_unit_sphere.e
        ))

    @staticmethod
    def random_in_unit_disk(size: int) -> cp.ndarray:
        r = cp.sqrt(random_float_list(size))
        theta = random_float_list(size) * 2 * cp.pi
        return cp.array([r*cp.cos(theta), r*cp.sin(theta)])


# Type aliases for Vec3
Point3 = Vec3  # 3D point
Color = Vec3  # RGB color


class Vec3List:
    def __init__(self, e: cp.ndarray):
        self.e = e

    def x(self) -> cp.ndarray:
        return cp.transpose(self.e)[0]

    def y(self) -> cp.ndarray:
        return cp.transpose(self.e)[1]

    def z(self) -> cp.ndarray:
        return cp.transpose(self.e)[2]

    def __getitem__(self, idx: int) -> Vec3:
        return Vec3(*self.e[idx])

    def get_ndarray(self, idx: int) -> cp.ndarray:
        return self.e[idx]

    def __setitem__(self, idx: int, val: Vec3) -> None:
        self.e[idx] = val.e

    def __len__(self) -> int:
        return len(self.e)

    def __add__(self, v: Union[Vec3, Vec3List]) -> Vec3List:
        return Vec3List(self.e + v.e)

    __radd__ = __add__

    def __iadd__(self, v: Union[Vec3, Vec3List]) -> Vec3List:
        self.e += v.e
        return self

    def __mul__(self, v: Union[Vec3, Vec3List, float]) -> Vec3List:
        if isinstance(v, (Vec3, Vec3List)):
            return Vec3List(self.e * v.e)
        # elif isinstance(v, (int, float, cp.floating)):
        return Vec3List(self.e * v)
        # raise TypeError

    __rmul__ = __mul__

    def __imul__(self, v: Union[Vec3, Vec3List, float]) -> Vec3List:
        if isinstance(v, (Vec3, Vec3List)):
            self.e *= v.e
        # elif isinstance(v, (int, float, cp.floating)):
        else:
            self.e *= v
            # raise TypeError
        return self

    def __sub__(self, v: Union[Vec3, Vec3List]) -> Vec3List:
        return Vec3List(self.e - v.e)

    def __rsub__(self, v: Union[Vec3, Vec3List]) -> Vec3List:
        return Vec3List(v.e - self.e)

    def __isub__(self, v: Union[Vec3, Vec3List]) -> Vec3List:
        self.e -= v.e
        return self

    def __neg__(self) -> Vec3List:
        return Vec3List(-self.e)

    def __truediv__(self, v: Union[Vec3List, float]) -> Vec3List:
        if isinstance(v, Vec3List):
            return Vec3List(self.e / v.e)
        # elif isinstance(v, (int, float, cp.floating)):
        return Vec3List(self.e / v)
        # raise TypeError

    def __matmul__(self, v: Vec3List) -> cp.ndarray:
        return (self.e * v.e).sum(axis=1)

    def mul_ndarray(self, a: cp.ndarray) -> Vec3List:
        return self * Vec3List.from_array(a)

    def div_ndarray(self, a: cp.ndarray) -> Vec3List:
        return self / Vec3List.from_array(a)

    def as_float32(self) -> Vec3List:
        self.e = self.e.astype(cp.float32, copy=False)
        return self

    def length_squared(self) -> cp.ndarray:
        return (self.e ** 2).sum(axis=1)

    def length(self) -> cp.ndarray:
        return cp.sqrt(self.length_squared())

    def unit_vector(self) -> Vec3List:
        length = self.length()
        condition = length > 0
        length_non_zero = cp.where(condition, length, 1)
        return self.div_ndarray(length_non_zero).mul_ndarray(condition)

    def reflect(self, n: Vec3List) -> Vec3List:
        return self - (n.mul_ndarray(self @ n)) * 2

    def refract(self, normal: Vec3List, etai_over_etat: cp.ndarray) \
            -> Vec3List:
        cos_theta = -self @ normal

        r_out_parallel = (
            self + normal.mul_ndarray(cos_theta)
        ).mul_ndarray(etai_over_etat)

        r_out_prep = normal.mul_ndarray(
            -cp.sqrt(1 - r_out_parallel.length_squared())
        )

        return r_out_parallel + r_out_prep

    @staticmethod
    def from_vec3(v: Vec3, length: int) -> Vec3List:
        vl = cp.tile(v.e, (length, 1))
        return Vec3List(vl)

    @staticmethod
    def from_array(a: cp.ndarray) -> Vec3List:
        vl = cp.transpose(cp.tile(a, (3, 1)))
        return Vec3List(vl)

    @staticmethod
    def new_empty(length: int) -> Vec3List:
        return Vec3List(cp.empty((length, 3), dtype=cp.float32))

    @staticmethod
    def new_zero(length: int) -> Vec3List:
        return Vec3List(cp.zeros((length, 3), dtype=cp.float32))
