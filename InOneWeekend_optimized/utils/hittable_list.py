import numpy as np  # type: ignore
from typing import List, Optional, Union
from utils.ray import RayList
from utils.hittable import Hittable, HitRecordList


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
            -> HitRecordList:
        if isinstance(t_max, (int, float, np.floating)):
            closest_so_far = np.full(len(r), t_max)
        else:
            closest_so_far = t_max

        rec = HitRecordList(
            np.empty((len(r), 3), dtype=np.float32),
            closest_so_far,
            [None for i in range(len(r))],
            np.empty((len(r), 3), dtype=np.float32),
            np.empty(len(r), dtype=np.bool)
        )
        for i, obj in enumerate(self.objects):
            temp_rec_list: HitRecordList = obj.hit(r, t_min, closest_so_far)
            rec.update(temp_rec_list)
            closest_so_far = rec.t

        return rec
