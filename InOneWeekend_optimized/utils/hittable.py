from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List, Union
from utils.vec3 import Vec3, Point3
from utils.ray import Ray, RayList

import typing
if typing.TYPE_CHECKING:
    from utils.material import Material


class HitRecord:
    def __init__(self, point: Point3, t: float, mat: Material,
                 normal: Vec3 = None, front_face: bool = None) -> None:
        self.p = point
        self.t = t
        self.material = mat
        self.normal = normal
        self.front_face = front_face

    def set_face_normal(self, r: Ray, outward_normal: Vec3) -> HitRecord:
        self.front_face = (r.direction() @ outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal
        return self


class HitRecordList:
    def __init__(self, point: np.ndarray, t: np.ndarray, mat: List[Material]) \
            -> None:
        self.p = point
        self.t = t
        self.material = mat
        self.normal: np.ndarray
        self.front_face: np.ndarray

    def set_face_normal(self, r: RayList, outward_normal: np.ndarray) \
            -> HitRecordList:
        self.front_face = (r.direction() * outward_normal).sum(axis=1) < 0
        self.normal = np.transpose(
            ((self.front_face + 1) * 2 - 3) * np.transpose(outward_normal)
        )
        return self

    def __getitem__(self, idx: int) -> HitRecord:
        return HitRecord(
            Point3(*self.p[idx]), self.t[idx], self.material[idx],
            Vec3(*self.normal[idx]), self.front_face[idx]
        )

    def __len__(self) -> int:
        return len(self.p)

    def __iter__(self) -> HitRecordList:
        self.idx = 0
        return self

    def __next__(self) -> HitRecord:
        if self.idx >= len(self):
            raise StopIteration
        result = self[self.idx]
        self.idx += 1
        return result

    def update(self, new: HitRecordList, max_list: np.ndarray) \
            -> HitRecordList:
        change = self.t < max_list
        change_3 = np.transpose(np.tile(change, (3, 1)))
        self.p = np.where(change_3, new.p, self.p)
        self.t = np.where(change, new.t, self.t)
        self.material = np.where(change, new.material, self.material)
        self.normal = np.where(change_3, new.normal, self.normal)
        self.front_face = np.where(change, new.front_face, self.front_face)
        return self


class Hittable(ABC):
    @abstractmethod
    def hit(self, r: RayList, t_min: float, t_max: Union[float, np.ndarray]) \
            -> HitRecordList:
        return NotImplemented
