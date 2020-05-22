import numpy as np  # type: ignore
import multiprocessing
import time
from joblib import Parallel, delayed  # type: ignore
from typing import List, Optional
from utils.vec3 import Vec3, Point3, Color
from utils.img import Img
from utils.ray import RayList
from utils.sphere import Sphere
from utils.hittable import Hittable, HitRecord
from utils.hittable_list import HittableList
from utils.rtweekend import random_float, random_float_list
from utils.camera import Camera
from utils.material import Lambertian, Metal, Dielectric


def ray_color(r: RayList, world: HittableList, depth: int) -> np.ndarray:
    if depth <= 0:
        return np.zeros((len(r), 3))
    result = np.empty((len(r), 3), dtype=np.float32)

    rec_list: List[Optional[HitRecord]] = world.hit(r, 0.001, np.inf)
    for i, rec in enumerate(rec_list):
        if rec is not None:
            scatter_result = rec.material.scatter(r[i], rec)
            if scatter_result is not None:
                scattered, attenuation = scatter_result
                result[i] = (
                    attenuation.e
                    * ray_color(RayList.single(scattered), world, depth-1)[0]
                )
            else:
                result[i] = Color(0, 0, 0).e
        else:
            unit_direction: Vec3 = r[i].direction().unit_vector()
            t = (unit_direction.y() + 1) * 0.5
            result[i] = (Color(1, 1, 1) * (1 - t) + Color(0.5, 0.7, 1) * t).e

    return result


def three_ball_scene() -> HittableList:
    world = HittableList()
    world.add(Sphere(
        Point3(0, 0, -1), 0.5, Lambertian(Color(0.1, 0.2, 0.5))
    ))
    world.add(Sphere(
        Point3(0, -100.5, -1), 100, Lambertian(Color(0.8, 0.8, 0))
    ))
    world.add(Sphere(
        Point3(1, 0, -1), 0.5, Metal(Color(0.8, 0.6, 0.2), 0.3)
    ))
    world.add(Sphere(
        Point3(-1, 0, -1), 0.5, Dielectric(1.5)
    ))
    world.add(Sphere(
        Point3(-1, 0, -1), -0.45, Dielectric(1.5)
    ))
    return world


def random_scene() -> HittableList:
    world = HittableList()

    ground_material = Lambertian(Color(0.5, 0.5, 0.5))
    world.add(Sphere(Point3(0, -1000, 0), 1000, ground_material))

    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = random_float()
            center = Point3(
                a + 0.9*random_float(), 0.2, b + 0.9*random_float()
            )

            if (center - Vec3(4, 0.2, 0)).length() > 0.9:
                if choose_mat < 0.3:
                    # Diffuse
                    albedo = Color.random() * Color.random()
                    sphere_material_diffuse = Lambertian(albedo)
                    world.add(Sphere(center, 0.2, sphere_material_diffuse))
                elif choose_mat < 0.7:
                    # Metal
                    albedo = Color.random(0.5, 1)
                    fuzz = random_float(0, 0.5)
                    sphere_material_metal = Metal(albedo, fuzz)
                    world.add(Sphere(center, 0.2, sphere_material_metal))
                else:
                    # Glass
                    sphere_material_glass = Dielectric(1.5)
                    world.add(Sphere(center, 0.2, sphere_material_glass))

    material_1 = Dielectric(1.5)
    world.add(Sphere(Point3(0, 1, 0), 1, material_1))

    material_2 = Lambertian(Color(0.4, 0.2, 0.1))
    world.add(Sphere(Point3(-4, 1, 0), 1, material_2))

    material_3 = Metal(Color(0.7, 0.6, 0.5), 0)
    world.add(Sphere(Point3(4, 1, 0), 1, material_3))

    return world


def scan_line(j: int, world: HittableList, cam: Camera,
              image_width: int, image_height: int,
              samples_per_pixel: int, max_depth: int) -> Img:
    img = Img(image_width, 1)
    row_pixel_color = np.tile(
        np.array([0, 0, 0], dtype=np.float32),
        (image_width, 1)
    )

    for s in range(samples_per_pixel):
        u: np.ndarray = (random_float_list(image_width)
                         + list(range(image_width))) / (image_width - 1)
        v: np.ndarray = (random_float_list(image_width)
                         + j) / (image_height - 1)
        r: RayList = cam.get_ray(u, v)
        row_pixel_color += ray_color(r, world, max_depth)

    img.write_pixel_list(0, row_pixel_color, samples_per_pixel)
    print(f"Scanlines remaining: {j} ", end="\r")
    return img


def main() -> None:
    aspect_ratio = 16 / 9
    image_width = 256
    image_height = int(image_width / aspect_ratio)
    samples_per_pixel = 20
    max_depth = 10

    world: HittableList = three_ball_scene()

    lookfrom = Point3(13, 2, 3)
    lookat = Point3(0, 0, 0)
    vup = Vec3(0, 1, 0)
    vfov = 20
    dist_to_focus: float = 10
    aperture: float = 0.1
    cam = Camera(
        lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus
    )

    print("Start rendering.")
    start_time = time.time()

    n_processer = multiprocessing.cpu_count()
    img_list: List[Img] = Parallel(n_jobs=n_processer)(
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
    final_img.save("./test.png", True)


if __name__ == "__main__":
    main()
