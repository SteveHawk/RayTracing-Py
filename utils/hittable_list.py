from typing import List, Optional
from utils.ray import Ray
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

    def hit(self, r: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        closest_so_far = t_max
        rec: Optional[HitRecord] = None

        for obj in self.objects:
            temp_rec = obj.hit(r, t_min, closest_so_far)
            if temp_rec is not None:
                closest_so_far = temp_rec.t
                rec = temp_rec

        return rec
