from typing import List, Optional
from utils.ray import Ray
from utils.hittable import Hittable, HitRecord
from utils.aabb import AABB


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

    def bounding_box(self, t0: float, t1: float) -> Optional[AABB]:
        if not self.objects:
            return None

        output_box = None
        for obj in self.objects:
            temp_box = obj.bounding_box(t0, t1)
            if temp_box is None:
                return None

            if output_box is None:
                output_box = temp_box
            else:
                output_box = AABB.surrounding_box(output_box, temp_box)

        return output_box
