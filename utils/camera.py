from utils.vec3 import Vec3, Point3
from utils.ray import Ray


class Camera:
    def __init__(self, aspect_ratio: float):
        self.origin = Point3(0, 0, 0)
        self.horizontal = Vec3(4, 0, 0)
        self.vertical = Vec3(0, 4/aspect_ratio, 0)
        self.distance = Vec3(0, 0, 1)
        self.lower_left_corner: Point3 = (
            self.origin - self.horizontal/2 - self.vertical/2 - self.distance
        )

    def get_ray(self, u: float, v: float) -> Ray:
        return Ray(
            self.origin,
            self.lower_left_corner + self.horizontal*u + self.vertical*v
        )
