import numpy as np  # type: ignore
import multiprocessing
import time
from joblib import Parallel, delayed  # type: ignore
from typing import List, Optional
import scenes
from utils.vec3 import Vec3, Point3, Color
from utils.img import Img
from utils.ray import Ray
from utils.hittable import HitRecord
from utils.rtweekend import random_float
from utils.camera import Camera
from utils.bvh import BVHNode


def ray_color(r: Ray, background: Color, world: BVHNode, depth: int) -> Color:
    # Bounce limit
    if depth <= 0:
        return Color(0, 0, 0)

    rec: Optional[HitRecord] = world.hit(r, 0.001, np.inf)

    # Ray hits nothing
    if rec is None:
        return background

    emitted = rec.material.emitted(rec.u, rec.v, rec.p)
    scatter_result = rec.material.scatter(r, rec)

    # No scattered ray (could be emissive material)
    if scatter_result is None:
        return emitted

    scattered, attenuation = scatter_result
    return (emitted + (
        attenuation * ray_color(scattered, background, world, depth-1)
    ))


def scan_line(j: int, background: Color, world: BVHNode, cam: Camera,
              image_width: int, image_height: int, samples_per_pixel: int,
              max_depth: int) -> Img:
    img = Img(image_width, 1)
    for i in range(image_width):
        pixel_color = Color(0, 0, 0)
        for s in range(samples_per_pixel):
            u: float = (i + random_float()) / (image_width - 1)
            v: float = (j + random_float()) / (image_height - 1)
            r: Ray = cam.get_ray(u, v)
            pixel_color += ray_color(r, background, world, max_depth)
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

    world, cam = scenes.simple_light(aspect_ratio, time0, time1)
    background = Color(0, 0, 0)

    print("Start rendering.")
    start_time = time.time()

    n_processer = multiprocessing.cpu_count()
    img_list: List[Img] = Parallel(n_jobs=n_processer, verbose=10)(
        delayed(scan_line)(
            j, background, world, cam,
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
