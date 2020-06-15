import numpy as np  # type: ignore
import multiprocessing
import time
from joblib import Parallel, delayed  # type: ignore
from typing import List, Optional
from scenes import random_scene, three_ball_scene, two_spheres
from utils.vec3 import Vec3, Point3, Color
from utils.img import Img
from utils.ray import Ray
from utils.hittable import HitRecord
from utils.rtweekend import random_float
from utils.camera import Camera
from utils.bvh import BVHNode


def ray_color(r: Ray, world: BVHNode, depth: int) -> Color:
    if depth <= 0:
        return Color(0, 0, 0)

    rec: Optional[HitRecord] = world.hit(r, 0.001, np.inf)
    if rec is not None:
        scatter_result = rec.material.scatter(r, rec)
        if scatter_result is not None:
            scattered, attenuation = scatter_result
            return attenuation * ray_color(scattered, world, depth-1)
        return Color(0, 0, 0)

    unit_direction: Vec3 = r.direction().unit_vector()
    t = (unit_direction.y() + 1) * 0.5
    return Color(1, 1, 1) * (1 - t) + Color(0.5, 0.7, 1) * t


def scan_line(j: int, world: BVHNode, cam: Camera,
              image_width: int, image_height: int,
              samples_per_pixel: int, max_depth: int) -> Img:
    img = Img(image_width, 1)
    for i in range(image_width):
        pixel_color = Color(0, 0, 0)
        for s in range(samples_per_pixel):
            u: float = (i + random_float()) / (image_width - 1)
            v: float = (j + random_float()) / (image_height - 1)
            r: Ray = cam.get_ray(u, v)
            pixel_color += ray_color(r, world, max_depth)
        img.write_pixel(i, 0, pixel_color, samples_per_pixel)
    print(f"Scanlines remaining: {j} ", end="\r")
    return img


def main() -> None:
    aspect_ratio = 16 / 9
    image_width = 256
    image_height = int(image_width / aspect_ratio)
    samples_per_pixel = 20
    max_depth = 10
    time0 = 0
    time1 = 1

    world, cam = two_spheres(aspect_ratio, time0, time1)

    print("Start rendering.")
    start_time = time.time()

    n_processer = multiprocessing.cpu_count()
    img_list: List[Img] = Parallel(n_jobs=n_processer, verbose=10)(
        delayed(scan_line)(
            j, world, cam,
            image_width, image_height,
            samples_per_pixel, max_depth
        ) for j in range(image_height-1, -1, -1)
    )

    final_img = Img(image_width, image_height)
    final_img.set_array(
        np.concatenate([img.frame for img in img_list])
    )

    end_time = time.time()
    print(f"\nDone. Total time: {round(end_time - start_time, 1)} s.")
    final_img.save("./output.png", True)


if __name__ == "__main__":
    main()
