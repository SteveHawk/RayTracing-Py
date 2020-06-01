from __future__ import annotations
import numpy as np  # type: ignore
from abc import ABC, abstractmethod
from typing import Optional, List, Union
from utils.vec3 import Vec3, Point3, Vec3List
from utils.ray import Ray, RayList

import typing
if typing.TYPE_CHECKING:
    from utils.material import Material


class HitRecord:
    def __init__(self, point: Point3, t: float, mat_idx: int,
                 normal: Vec3 = Vec3(), front_face: bool = False) -> None:
        self.p = point
        self.t = t
        self.material_idx = mat_idx
        self.normal = normal
        self.front_face = front_face

    def set_face_normal(self, r: Ray, outward_normal: Vec3) -> HitRecord:
        self.front_face = (r.direction() @ outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal
        return self


class HitRecordList:
    def __init__(self, point: Vec3List, t: np.ndarray, mat: np.ndarray,
                 normal: Vec3List = Vec3List.new_zero(0),
                 front_face: np.ndarray = np.array([])) -> None:
        self.p = point
        self.t = t
        self.material = mat
        self.normal = normal
        self.front_face = front_face

    def set_face_normal(self, r: RayList, outward_normal: Vec3List) \
            -> HitRecordList:
        self.front_face = (r.direction() * outward_normal).e.sum(axis=1) < 0
        front_face_3 = Vec3List.from_array(self.front_face)
        self.normal = Vec3List(
            np.where(front_face_3.e, outward_normal.e, -outward_normal.e)
        )
        return self

    def __getitem__(self, idx: int) -> Optional[HitRecord]:
        mat_idx: int = self.material[idx]
        if mat_idx == 0:
            return None
        else:
            return HitRecord(
                self.p[idx], self.t[idx], mat_idx,
                self.normal[idx], self.front_face[idx]
            )

    def __setitem__(self, idx: int, rec: HitRecord) -> None:
        self.p[idx] = rec.p
        self.t[idx] = rec.t
        self.material[idx] = rec.material_idx
        self.normal[idx] = rec.normal
        self.front_face[idx] = rec.front_face

    def __len__(self) -> int:
        return len(self.t)

    def __iter__(self) -> HitRecordList:
        self.idx = 0
        return self

    def __next__(self) -> Optional[HitRecord]:
        if self.idx >= len(self):
            raise StopIteration
        result = self[self.idx]
        self.idx += 1
        return result

    def set_compress_info(self, idx: Optional[np.ndarray]) -> HitRecordList:
        self.compress_idx = idx
        return self

    def update(self, new: HitRecordList) -> HitRecordList:
        if new.compress_idx is not None:
            idx = new.compress_idx
            old_idx = np.arange(len(idx))
        else:
            idx = slice(0, -1)
            old_idx = slice(0, -1)

        change = (new.t[old_idx] < self.t[idx]) & (new.t[old_idx] > 0)
        if not change.any():
            return self
        change_3 = Vec3List.from_array(change)

        self.p.e[idx] = np.where(
            change_3.e, new.p.e[old_idx], self.p.e[idx]
        )
        self.t[idx] = np.where(
            change, new.t[old_idx], self.t[idx]
        )
        self.material[idx] = np.where(
            change, new.material[old_idx], self.material[idx]
        )
        self.normal.e[idx] = np.where(
            change_3.e, new.normal.e[old_idx], self.normal.e[idx]
        )
        self.front_face[idx] = np.where(
            change, new.front_face[old_idx], self.front_face[idx]
        )

        return self

    @staticmethod
    def new(length: int) -> HitRecordList:
        return HitRecordList(
            Vec3List.new_empty(length),
            np.zeros(length, dtype=np.float32),
            np.zeros(length, dtype=np.int32),
            Vec3List.new_empty(length),
            np.empty(length, dtype=np.bool)
        )

    @staticmethod
    def new_from_t(t: np.ndarray) -> HitRecordList:
        length = len(t)
        return HitRecordList(
            Vec3List.new_empty(length),
            t,
            np.zeros(length, dtype=np.int32),
            Vec3List.new_empty(length),
            np.empty(length, dtype=np.bool)
        )


class Hittable(ABC):
    @abstractmethod
    def __init__(self):
        self.material: Material

    @abstractmethod
    def hit(self, r: RayList, t_min: float, t_max: Union[float, np.ndarray]) \
            -> HitRecordList:
        return NotImplemented
