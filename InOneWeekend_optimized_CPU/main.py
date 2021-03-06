import numpy as np  # type: ignore
import multiprocessing
import time
from joblib import Parallel, delayed  # type: ignore
from typing import List, Optional, Dict, Tuple
from utils.vec3 import Vec3, Point3, Color, Vec3List
from utils.img import Img
from utils.ray import RayList, Ray
from utils.sphere import Sphere
from utils.hittable import Hittable, HitRecordList, HitRecord
from utils.hittable_list import HittableList
from utils.rtweekend import random_float, random_float_list
from utils.camera import Camera
from utils.material import Material, Lambertian, Metal, Dielectric


def three_ball_scene() -> HittableList:
    world = HittableList()
    world.add(Sphere(
        Point3(0, 0, -1), 0.5, Lambertian(Color(0.1, 0.2, 0.5), 1)
    ))
    world.add(Sphere(
        Point3(0, -100.5, -1), 100, Lambertian(Color(0.8, 0.8, 0), 2)
    ))
    world.add(Sphere(
        Point3(1, 0, -1), 0.5, Metal(Color(0.8, 0.6, 0.2), 0.3, 3)
    ))
    material_dielectric = Dielectric(1.5, 4)
    world.add(Sphere(
        Point3(-1, 0, -1), 0.5, material_dielectric
    ))
    world.add(Sphere(
        Point3(-1, 0, -1), -0.45, material_dielectric
    ))
    return world


def random_scene() -> HittableList:
    world = HittableList()

    ground_material = Lambertian(Color(0.5, 0.5, 0.5), 1)
    world.add(Sphere(Point3(0, -1000, 0), 1000, ground_material))

    sphere_material_glass = Dielectric(1.5, 2)
    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = random_float()
            center = Point3(
                a + 0.9*random_float(), 0.2, b + 0.9*random_float()
            )

            if (center - Vec3(4, 0.2, 0)).length() > 0.9:
                idx = (a*22 + b) + (11*22 + 11) + 6
                if choose_mat < 0.6:
                    # Diffuse
                    albedo = Color.random() * Color.random()
                    sphere_material_diffuse = Lambertian(albedo, idx)
                    world.add(Sphere(center, 0.2, sphere_material_diffuse))
                elif choose_mat < 0.8:
                    # Metal
                    albedo = Color.random(0.5, 1)
                    fuzz = random_float(0, 0.5)
                    sphere_material_metal = Metal(albedo, fuzz, idx)
                    world.add(Sphere(center, 0.2, sphere_material_metal))
                else:
                    # Glass
                    world.add(Sphere(center, 0.2, sphere_material_glass))

    material_1 = Dielectric(1.5, 3)
    world.add(Sphere(Point3(0, 1, 0), 1, material_1))

    material_2 = Lambertian(Color(0.4, 0.2, 0.1), 4)
    world.add(Sphere(Point3(-4, 1, 0), 1, material_2))

    material_3 = Metal(Color(0.7, 0.6, 0.5), 0, 5)
    world.add(Sphere(Point3(4, 1, 0), 1, material_3))

    return world


def compress(r: RayList, rec: HitRecordList) \
        -> Tuple[RayList, HitRecordList, Optional[np.ndarray]]:
    condition = rec.t > 0
    full_rate = condition.sum() / len(r)
    if full_rate > 0.6:
        return r, rec, None

    idx: np.ndarray = np.where(condition)[0]
    new_r = RayList(
        Vec3List(r.orig.get_ndarray(idx)), Vec3List(r.dir.get_ndarray(idx))
    )
    new_rec = HitRecordList(
        Vec3List(rec.p.get_ndarray(idx)),
        rec.t[idx],
        rec.material[idx],
        Vec3List(rec.normal.get_ndarray(idx)),
        rec.front_face[idx]
    )
    return new_r, new_rec, idx


def decompress(r: RayList, a: Vec3List, idx: Optional[np.ndarray],
               length: int) -> Tuple[RayList, Vec3List]:
    if idx is None:
        return r, a

    old_idx = np.arange(len(idx))

    new_r = RayList.new_zero(length)
    new_r.orig.e[idx] = r.orig.e[old_idx]
    new_r.dir.e[idx] = r.dir.e[old_idx]

    new_a = Vec3List.new_zero(length)
    new_a.e[idx] = a.e[old_idx]

    return new_r, new_a


def ray_color(r: RayList, world: HittableList, depth: int) -> Vec3List:
    length = len(r)
    if not r.direction().e.any():
        return Vec3List.new_zero(length)

    # Calculate object hits
    rec_list: HitRecordList = world.hit(r, 0.001, np.inf)

    # Useful empty arrays
    empty_vec3list = Vec3List.new_zero(length)
    empty_array_float = np.zeros(length, np.float32)
    empty_array_bool = np.zeros(length, np.bool)
    empty_array_int = np.zeros(length, np.int32)

    # Background / Sky
    unit_direction = r.direction().unit_vector()
    sky_condition = Vec3List.from_array(
        (unit_direction.length() > 0) & (rec_list.material == 0)
    )
    t = (unit_direction.y() + 1) * 0.5
    blue_bg = (
        Vec3List.from_vec3(Color(1, 1, 1), length).mul_ndarray(1 - t)
        + Vec3List.from_vec3(Color(0.5, 0.7, 1), length).mul_ndarray(t)
    )
    result_bg = Vec3List(
        np.where(sky_condition.e, blue_bg.e, empty_vec3list.e)
    )
    if depth <= 1:
        return result_bg

    # Per-material preparations
    materials: Dict[int, Material] = world.get_materials()
    material_dict: Dict[int, Tuple[RayList, HitRecordList]] = dict()
    for mat_idx in materials:
        mat_condition = (rec_list.material == mat_idx)
        mat_condition_3 = Vec3List.from_array(mat_condition)
        if not mat_condition.any():
            continue
        raylist_temp = RayList(
            Vec3List(np.where(mat_condition_3.e, r.orig.e, empty_vec3list.e)),
            Vec3List(np.where(mat_condition_3.e, r.dir.e, empty_vec3list.e))
        )
        reclist_temp = HitRecordList(
            Vec3List(np.where(
                mat_condition_3.e, rec_list.p.e, empty_vec3list.e
            )),
            np.where(mat_condition, rec_list.t, empty_array_float),
            np.where(mat_condition, rec_list.material, empty_array_int),
            Vec3List(np.where(
                mat_condition_3.e, rec_list.normal.e, empty_vec3list.e
            )),
            np.where(mat_condition, rec_list.front_face, empty_array_bool)
        )
        material_dict[mat_idx] = raylist_temp, reclist_temp

    # Material scatter calculations
    scattered_list = RayList.new_zero(length)
    attenuation_list = Vec3List.new_zero(length)
    for key in material_dict:
        ray, rec = material_dict[key]
        ray, rec, idx_list = compress(ray, rec)

        scattered, attenuation = materials[key].scatter(ray, rec)
        scattered, attenuation = decompress(
            scattered, attenuation, idx_list, length
        )
        scattered_list += scattered
        attenuation_list += attenuation
    result_hittable = (
        attenuation_list * ray_color(scattered_list, world, depth-1)
    )

    return result_hittable + result_bg


def scan_line(j: int, world: HittableList, cam: Camera,
              image_width: int, image_height: int,
              samples_per_pixel: int, max_depth: int) -> Img:
    img = Img(image_width, 1)
    row_pixel_color = Vec3List.from_vec3(Color(), image_width)

    for s in range(samples_per_pixel):
        u: np.ndarray = (random_float_list(image_width)
                         + np.arange(image_width)) / (image_width - 1)
        v: np.ndarray = (random_float_list(image_width)
                         + j) / (image_height - 1)
        r: RayList = cam.get_ray(u, v)
        row_pixel_color += ray_color(r, world, max_depth)

    img.write_pixel_list(0, row_pixel_color, samples_per_pixel)
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
    img_list: List[Img] = Parallel(n_jobs=n_processer, verbose=10)(
        delayed(scan_line)(
            j, world, cam,
            image_width, image_height,
            samples_per_pixel, max_depth
        ) for j in range(image_height-1, -1, -1)
    )

    # # Profile prologue
    # import cProfile
    # import pstats
    # import io
    # from pstats import SortKey
    # pr = cProfile.Profile()
    # pr.enable()

    # img_list: List[Img] = list()
    # for j in range(image_height-1, -1, -1):
    #     img_list.append(
    #         scan_line(
    #             j, world, cam,
    #             image_width, image_height,
    #             samples_per_pixel, max_depth
    #         )
    #     )

    # # Profile epilogue
    # pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())

    final_img = Img(image_width, image_height)
    final_img.set_array(
        np.concatenate([img.frame for img in img_list])
    )

    end_time = time.time()
    print(f"\nDone. Total time: {round(end_time - start_time, 1)} s.")
    final_img.save("./output.png", True)


if __name__ == "__main__":
    main()
