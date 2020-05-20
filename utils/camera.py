import numpy as np  # type: ignore
from utils.vec3 import Vec3, Point3
from utils.ray import Ray
from utils.rtweekend import degrees_to_radians


class Camera:
    def __init__(self, lookfrom: Point3, lookat: Point3, vup: Vec3,
                 vfov: float, aspect_ratio: float) -> None:
        """
        vfov: vertical field-of-view in degress
        """
        theta: float = degrees_to_radians(vfov)
        h: float = np.tan(theta / 2)
        viewport_height: float = 2 * h
        viewport_width: float = aspect_ratio * viewport_height

        w: Vec3 = (lookfrom - lookat).unit_vector()
        u: Vec3 = vup.cross(w).unit_vector()
        v: Vec3 = w.cross(u)

        self.origin: Point3 = lookfrom
        self.horizontal: Vec3 = u * viewport_width
        self.vertical: Vec3 = v * viewport_height
        self.lower_left_corner: Point3 = (
            self.origin - self.horizontal/2 - self.vertical/2 - w
        )

    def get_ray(self, u: float, v: float) -> Ray:
        return Ray(
            self.origin, (self.lower_left_corner
                          + self.horizontal*u + self.vertical*v
                          - self.origin)
        )
