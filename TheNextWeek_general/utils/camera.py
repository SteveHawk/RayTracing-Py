import numpy as np  # type: ignore
from utils.vec3 import Vec3, Point3
from utils.ray import Ray
from utils.rtweekend import degrees_to_radians


class Camera:
    def __init__(self, lookfrom: Point3, lookat: Point3, vup: Vec3,
                 vfov: float, aspect_ratio: float,
                 aperture: float, focus_dist: float) -> None:
        """
        vfov: vertical field-of-view in degress
        """
        theta: float = degrees_to_radians(vfov)
        h: float = np.tan(theta / 2)
        viewport_height: float = 2 * h
        viewport_width: float = aspect_ratio * viewport_height

        self.w: Vec3 = (lookfrom - lookat).unit_vector()
        self.u: Vec3 = vup.cross(self.w).unit_vector()
        self.v: Vec3 = self.w.cross(self.u)

        self.origin: Point3 = lookfrom
        self.horizontal: Vec3 = self.u * viewport_width * focus_dist
        self.vertical: Vec3 = self.v * viewport_height * focus_dist
        self.lower_left_corner: Point3 = (
            self.origin - self.horizontal/2 - self.vertical/2
            - self.w * focus_dist
        )
        self.lens_radius: float = aperture / 2

    def get_ray(self, s: float, t: float) -> Ray:
        rd: Vec3 = Vec3.random_in_unit_disk() * self.lens_radius
        offset: Vec3 = self.u * rd.x() + self.v * rd.y()

        return Ray(
            self.origin + offset,
            (self.lower_left_corner
             + self.horizontal*s + self.vertical*t
             - self.origin - offset)
        )
