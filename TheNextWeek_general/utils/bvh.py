from typing import Optional, List, Callable
from utils.hittable import Hittable, HitRecord
from utils.ray import Ray
from utils.aabb import AABB
from utils.hittable_list import HittableList
from utils.rtweekend import random_int


class BVHNode(Hittable):
    def __init__(self, objects: List[Hittable], time0: float, time1: float):
        axis = random_int(0, 2)
        key_func: Callable[[Hittable], float] = self.key_func(axis)

        length = len(objects)
        if length == 1:
            self.left = self.right = objects[0]
        elif length == 2:
            if key_func(objects[0]) < key_func(objects[1]):
                self.left = objects[0]
                self.right = objects[1]
            else:
                self.left = objects[1]
                self.right = objects[0]
        else:
            sorted_objects = sorted(objects, key=key_func)
            mid = int(length / 2)
            self.left = BVHNode(sorted_objects[0:mid], time0, time1)
            self.right = BVHNode(sorted_objects[mid:length], time0, time1)

        box_left = self.left.bounding_box(time0, time1)
        box_right = self.right.bounding_box(time0, time1)
        if box_left is None or box_right is None:
            print("No bounding box in bvh_node constructor.")
            raise ValueError
        self.box = AABB.surrounding_box(box_left, box_right)

    def hit(self, r: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        if not self.box.hit(r, t_min, t_max):
            return None

        rec_l = self.left.hit(r, t_min, t_max)
        rec_r = self.right.hit(r, t_min, t_max if rec_l is None else rec_l.t)

        if rec_r is not None:
            return rec_r
        return rec_l

    def bounding_box(self, t0: float, t1: float) -> Optional[AABB]:
        return self.box

    @staticmethod
    def key_func(axis: int) -> Callable[[Hittable], float]:

        def key(h: Hittable) -> float:
            box: Optional[AABB] = h.bounding_box(0, 0)
            if box is None:
                print("No bounding box in bvh_node constructor.")
                raise ValueError
            return box.min().e[axis]

        return key
