import numpy as np  # type: ignore
from typing import List, Optional, Union, Tuple
from utils.vec3 import Vec3List
from utils.ray import RayList
from utils.hittable import Hittable, HitRecordList


class HittableList(Hittable):
    def __init__(self, obj: Optional[Hittable] = None) -> None:
        self.objects: List[Hittable] = list()
        if obj is not None:
            self.add(obj)

    def add(self, obj: Hittable) -> None:
        self.objects.append(obj)

    def clear(self) -> None:
        self.objects.clear()

    def hit(self, r: RayList, t_min: float, t_max: Union[float, np.ndarray]) \
            -> HitRecordList:
        if isinstance(t_max, (int, float, np.floating)):
            closest_so_far = np.full(len(r), t_max)
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
        full_rate = condition.sum()
        if full_rate > 0.3:
            self.idx = None
            return r, closest_so_far

        self.idx = np.where(condition)[0]
        new_r = RayList(
            r.orig.get_ndarray(self.idx),
            r.dir.get_ndarray(self.idx)
        )
        new_c = closest_so_far[self.idx]
        return new_r, new_c

    def decompress(self, rec: HitRecordList) -> HitRecordList:
        if self.idx is None:
            return rec
        old_idx = np.arange(len(self.idx))
        new_rec = HitRecordList.new(len(self.idx))
        new_rec.p.e[self.idx] = rec.p.e[old_idx]
        new_rec.t[self.idx] = rec.t[old_idx]
        new_rec.material[self.idx] = rec.material[old_idx]
        new_rec.normal.e[self.idx] = rec.normal.e[old_idx]
        new_rec.front_face[self.idx] = rec.front_face[old_idx]
        return new_rec
