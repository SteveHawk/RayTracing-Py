import numpy as np  # type: ignore
from typing import List, Optional, Union
from utils.ray import RayList
from utils.hittable import Hittable, HitRecord


class HittableList(Hittable):
    def __init__(self, obj: Optional[Hittable] = None):
        self.objects: List[Hittable] = list()
        if obj is not None:
            self.add(obj)

    def add(self, obj: Hittable):
        self.objects.append(obj)

    def clear(self):
        self.objects.clear()

    def hit(self, r: RayList, t_min: float, t_max: Union[float, np.ndarray]) \
            -> List[Optional[HitRecord]]:
        if isinstance(t_max, (int, float, np.floating)):
            closest_so_far = np.full(len(r), t_max)
        else:
            closest_so_far = t_max

        rec: List[Optional[HitRecord]] = [None for i in range(len(r))]

        for obj in self.objects:
            temp_rec_list = obj.hit(r, t_min, closest_so_far)
            for i, temp_rec in enumerate(temp_rec_list):
                if temp_rec is not None:
                    closest_so_far[i] = temp_rec.t
                    rec[i] = temp_rec

        return rec
