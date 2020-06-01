import cupy as cp  # type: ignore
from typing import List
from utils.vec3 import Vec3, Point3, Vec3List
from utils.ray import RayList
from utils.rtweekend import degrees_to_radians


class Camera:
    def __init__(self, lookfrom: Point3, lookat: Point3, vup: Vec3,
                 vfov: float, aspect_ratio: float,
                 aperture: float, focus_dist: float) -> None:
        """
        vfov: vertical field-of-view in degress
        """
        theta: float = degrees_to_radians(vfov)
        h: float = cp.tan(theta / 2)
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

    def get_ray(self, s: cp.ndarray, t: cp.ndarray) -> RayList:
        if len(s) != len(t):
            raise ValueError
        rd: cp.ndarray = Vec3.random_in_unit_disk(len(s)) * self.lens_radius

        u_multi = Vec3List.from_vec3(self.u, len(s))
        v_multi = Vec3List.from_vec3(self.v, len(s))
        offset_list = u_multi.mul_ndarray(rd[0]) + v_multi.mul_ndarray(rd[1])

        origin_list = offset_list + self.origin

        horizontal_multi = Vec3List.from_vec3(self.horizontal, len(s))
        vertical_multi = Vec3List.from_vec3(self.vertical, len(s))
        direction_list = (
            horizontal_multi.mul_ndarray(s) + vertical_multi.mul_ndarray(t)
            + self.lower_left_corner - self.origin - offset_list
        )

        return RayList(origin_list, direction_list)
