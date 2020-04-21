from utils.vec3 import Vec3


class Ray:
    def __init__(self, origin: Vec3 = Vec3(),
                 direction: Vec3 = Vec3()) -> None:
        self.orig = origin
        self.dir = direction

    def origin(self) -> Vec3:
        return self.orig

    def direction(self) -> Vec3:
        return self.dir

    def at(self, t: float) -> Vec3:
        return self.orig + self.dir * t
