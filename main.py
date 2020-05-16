import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from typing import List
from utils.vec3 import Vec3, Point3, Color
from utils.img import Img
from utils.ray import Ray


def hit_sphere(center: Point3, radius: float, r: Ray) -> float:
    oc: Vec3 = r.origin() - center
    a: float = r.direction().length_squared()
    half_b: float = oc @ r.direction()
    c: float = oc.length_squared() - radius**2
    discriminant: float = half_b**2 - a*c
    if discriminant < 0:
        return -1
    else:
        return (-half_b - np.sqrt(discriminant)) / a


def ray_color(r: Ray) -> Color:
    center = Point3(0, 0, -1)
    t: float = hit_sphere(center, 0.5, r)
    if t > 0:
        N: Vec3 = (r.at(t) - center).unit_vector()
        return Color(N.x() + 1, N.y() + 1, N.z() + 1) * 0.5
    unit_direction: Vec3 = r.direction().unit_vector()
    t = (unit_direction.y() + 1) * 0.5
    return Color(1, 1, 1) * (1 - t) + Color(0.5, 0.7, 1) * t


def write_pic(j: int, image_width: int, image_height: int,
              origin: Point3, lower_left_corner: Point3,
              horizontal: Vec3, vertical: Vec3) -> Img:
    img = Img(image_width, 1)
    for i in range(image_width):
        u = i / image_width
        v = j / image_height
        r = Ray(origin, lower_left_corner + horizontal*u + vertical*v)
        color = ray_color(r)
        img.write_pixel(i, 0, color)
    print(f"Scanlines remaining: {j} ", end="\r")
    return img


def main() -> None:
    aspect_ratio = 16 / 9
    image_width = 640
    image_height = int(image_width / aspect_ratio)
    final_img = Img(image_width, image_height)

    origin = Point3(0, 0, 0)
    horizontal = Vec3(4, 0, 0)
    vertical = Vec3(0, 2.25, 0)
    distance = Vec3(0, 0, 1)
    lower_left_corner: Point3 = origin - horizontal/2 - vertical/2 - distance

    n_processer = multiprocessing.cpu_count()
    img_list: List[Img] = Parallel(n_jobs=n_processer)(
        delayed(write_pic)(
            j, image_width, image_height,
            origin, lower_left_corner,
            horizontal, vertical
        ) for j in range(image_height-1, -1, -1)
    )
    final_img.set_array(
        np.concatenate([img.frame for img in img_list])
    )

    print("\nDone.")
    final_img.save("./test.png", True)


if __name__ == "__main__":
    main()
