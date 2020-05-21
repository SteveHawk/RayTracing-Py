import numpy as np  # type: ignore
from typing import List
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

    def get_ray(self, s: np.ndarray, t: np.ndarray) -> List[Ray]:
        if len(s) != len(t):
            raise ValueError
        rd: np.ndarray = Vec3.random_in_unit_disk(len(s)) * self.lens_radius

        u_multi = np.tile(self.u.e, (len(s), 1))
        v_multi = np.tile(self.v.e, (len(s), 1))
        offset_list = (
            np.transpose(rd[0] * np.transpose(u_multi))
            + np.transpose(rd[1] * np.transpose(v_multi))
        )

        origin_list = self.origin.e + offset_list

        horizontal_multi = np.tile(self.horizontal.e, (len(s), 1))
        vertical_multi = np.tile(self.vertical.e, (len(s), 1))
        direction_list = (
            self.lower_left_corner.e
            + np.transpose(np.transpose(horizontal_multi) * s)
            + np.transpose(np.transpose(vertical_multi) * t)
            - self.origin.e - offset_list
        )

        return [Ray(Point3(*origin), Vec3(*direction))
                for origin, direction in zip(origin_list, direction_list)]
