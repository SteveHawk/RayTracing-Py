from __future__ import annotations
from utils.vec3 import Vec3, Point3
from utils.ray import Ray


class AABB:
    def __init__(self, _min: Point3 = Point3(),
                 _max: Point3 = Point3()) -> None:
        self._min = _min
        self._max = _max

    def min(self) -> Point3:
        return self._min

    def max(self) -> Point3:
        return self._max

    def hit_deprecated(self, r: Ray, tmin: float, tmax: float) -> bool:
        for i in range(3):
            t0: float = min(
                (self.min()[i] - r.origin()[i]) / r.direction()[i],
                (self.max()[i] - r.origin()[i]) / r.direction()[i]
            )
            t1: float = min(
                (self.min()[i] - r.origin()[i]) / r.direction()[i],
                (self.max()[i] - r.origin()[i]) / r.direction()[i]
            )
            tmin = max(t0, tmin)
            tmax = min(t1, tmax)
            if tmax <= tmin:
                return False
        return True

    def hit(self, r: Ray, tmin: float, tmax: float) -> bool:
        for i in range(3):
            invD: float = 1 / r.direction()[i]
            t0: float = (self.min()[i] - r.origin()[i]) * invD
            t1: float = (self.max()[i] - r.origin()[i]) * invD
            if invD < 0:
                t0, t1 = t1, t0
            tmin = max(t0, tmin)
            tmax = min(t1, tmax)
            if tmax <= tmin:
                return False
        return True

    @staticmethod
    def surrounding_box(box0: AABB, box1: AABB) -> AABB:
        small = Point3(
            min(box0.min().x(), box1.min().x()),
            min(box0.min().y(), box1.min().y()),
            min(box0.min().z(), box1.min().z())
        )
        big = Point3(
            max(box0.max().x(), box1.max().x()),
            max(box0.max().y(), box1.max().y()),
            max(box0.max().z(), box1.max().z())
        )
        return AABB(small, big)
