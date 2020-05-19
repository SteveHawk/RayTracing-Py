import numpy as np  # type: ignore
from utils.vec3 import Vec3, Point3
from utils.ray import Ray
from utils.rtweekend import degrees_to_radians


class Camera:
    def __init__(self, vfov: float, aspect_ratio: float) -> None:
        """
        vfov: vertical field-of-view in degress
        """
        theta: float = degrees_to_radians(vfov)
        h: float = np.tan(theta / 2)
        viewport_height: float = 2 * h
        viewport_width: float = aspect_ratio * viewport_height
        focal_length: float = 1

        self.origin = Point3(0, 0, 0)
        self.horizontal = Vec3(viewport_width, 0, 0)
        self.vertical = Vec3(0, viewport_height, 0)
        self.distance = Vec3(0, 0, focal_length)
        self.lower_left_corner: Point3 = (
            self.origin - self.horizontal/2 - self.vertical/2 - self.distance
        )

    def get_ray(self, u: float, v: float) -> Ray:
        return Ray(
            self.origin,
            self.lower_left_corner + self.horizontal*u + self.vertical*v
        )
