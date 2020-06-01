import numpy as np  # type: ignore
from typing import List, Optional, Union, Tuple, Dict
from utils.vec3 import Vec3List
from utils.ray import RayList
from utils.material import Material
from utils.hittable import Hittable, HitRecordList


class HittableList(Hittable):
    def __init__(self, obj: Optional[Hittable] = None) -> None:
        self.objects: List[Hittable] = list()
        self.materials: Dict[int, Material] = dict()
        if obj is not None:
            self.add(obj)

    def add(self, obj: Hittable) -> None:
        self.objects.append(obj)
        if obj.material.idx < 0:
            raise ValueError
        if obj.material.idx not in self.materials:
            self.materials[obj.material.idx] = obj.material

    def clear(self) -> None:
        self.objects.clear()

    def get_materials(self) -> Dict[int, Material]:
        return self.materials

    def hit(self, r: RayList, t_min: float, t_max: Union[float, np.ndarray]) \
            -> HitRecordList:
        if isinstance(t_max, (int, float, np.floating)):
            closest_so_far = np.full(len(r), t_max, dtype=np.float32)
        else:
            closest_so_far = t_max

        r, closest_so_far = self.compress(r, closest_so_far)

        rec = HitRecordList.new_from_t(closest_so_far)
        for obj in self.objects:
            temp_rec_list: HitRecordList = obj.hit(r, t_min, closest_so_far)
            rec.update(temp_rec_list)
            closest_so_far = rec.t

        return self.decompress(rec)

    def compress(self, r: RayList, closest_so_far: np.ndarray) \
            -> Tuple[RayList, np.ndarray]:
        condition = r.dir.length_squared() > 0
        full_rate = condition.sum() / len(r)
        if full_rate > 0.6:
            self.idx = None
            return r, closest_so_far

        self.idx = np.where(condition)[0]
        self.old_length = len(condition)
        new_r = RayList(
            Vec3List(r.orig.get_ndarray(self.idx)),
            Vec3List(r.dir.get_ndarray(self.idx))
        )
        new_c = closest_so_far[self.idx]
        return new_r, new_c

    def decompress(self, rec: HitRecordList) -> HitRecordList:
        if self.idx is None:
            return rec
        old_idx = np.arange(len(self.idx))
        new_rec = HitRecordList.new(self.old_length)
        new_rec.p.e[self.idx] = rec.p.e[old_idx]
        new_rec.t[self.idx] = rec.t[old_idx]
        new_rec.material[self.idx] = rec.material[old_idx]
        new_rec.normal.e[self.idx] = rec.normal.e[old_idx]
        new_rec.front_face[self.idx] = rec.front_face[old_idx]
        return new_rec
