import numpy as np  # type: ignore
import multiprocessing
from joblib import Parallel, delayed  # type: ignore
from typing import List
from utils.vec3 import Vec3, Point3, Color
from utils.img import Img
from utils.ray import Ray
from utils.sphere import Sphere
from utils.hittable import Hittable, HitRecord
from utils.hittable_list import HittableList
from utils.rtweekend import degrees_to_radians


def ray_color(r: Ray, world: Hittable) -> Color:
    rec = world.hit(r, 0, np.inf)
    if rec is not None:
        return (rec.normal + Color(1, 1, 1)) * 0.5
    unit_direction: Vec3 = r.direction().unit_vector()
    t = (unit_direction.y() + 1) * 0.5
    return Color(1, 1, 1) * (1 - t) + Color(0.5, 0.7, 1) * t


def write_pic(j: int, world: HittableList,
              image_width: int, image_height: int,
              origin: Point3, lower_left_corner: Point3,
              horizontal: Vec3, vertical: Vec3) -> Img:
    img = Img(image_width, 1)
    for i in range(image_width):
        u = i / image_width
        v = j / image_height
        r = Ray(origin, lower_left_corner + horizontal*u + vertical*v)
        color = ray_color(r, world)
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
    vertical = Vec3(0, 4/aspect_ratio, 0)
    distance = Vec3(0, 0, 1)
    lower_left_corner: Point3 = origin - horizontal/2 - vertical/2 - distance

    world: HittableList = HittableList()
    world.add(Sphere(Point3(0, 0, -1), 0.5))
    world.add(Sphere(Point3(0, -100.5, -1), 100))

    n_processer = multiprocessing.cpu_count()
    img_list: List[Img] = Parallel(n_jobs=n_processer)(
        delayed(write_pic)(
            j, world,
            image_width, image_height,
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
