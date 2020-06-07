from utils.vec3 import Vec3, Point3


class Ray:
    def __init__(self, origin: Point3 = Point3(),
                 direction: Vec3 = Vec3(),
                 time: float = 0) -> None:
        self.orig = origin
        self.dir = direction
        self.tm = time

    def origin(self) -> Point3:
        return self.orig

    def direction(self) -> Vec3:
        return self.dir

    def time(self) -> float:
        return self.tm

    def at(self, t: float) -> Point3:
        return self.orig + self.dir * t
