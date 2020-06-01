import cupy as cp  # type: ignore
import multiprocessing
import time
import os
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
        -> Tuple[RayList, HitRecordList, Optional[cp.ndarray]]:
    condition = rec.t > 0
    full_rate = condition.sum() / len(r)
    if full_rate > 0.5:
        return r, rec, None

    idx: cp.ndarray = cp.where(condition)[0]
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


def decompress(r: RayList, a: Vec3List, idx: Optional[cp.ndarray],
               length: int) -> Tuple[RayList, Vec3List]:
    if idx is None:
        return r, a

    old_idx = cp.arange(len(idx))

    new_r = RayList.new_zero(length)
    new_r.orig.e[idx] = r.orig.e[old_idx]
    new_r.dir.e[idx] = r.dir.e[old_idx]

    new_a = Vec3List.new_zero(length)
    new_a.e[idx] = a.e[old_idx]

    return new_r, new_a


def ray_color(r: RayList, world: HittableList, depth: int) \
        -> Tuple[Optional[RayList], Optional[Vec3List], Vec3List]:
    length = len(r)
    if not r.direction().e.any():
        return None, None, Vec3List.new_zero(length)

    # Calculate object hits
    rec_list: HitRecordList = world.hit(r, 0.001, cp.inf)

    # Useful empty arrays
    empty_vec3list = Vec3List.new_zero(length)
    empty_array_float = cp.zeros(length, cp.float32)
    empty_array_bool = cp.zeros(length, cp.bool)
    empty_array_int = cp.zeros(length, cp.int32)

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
        cp.where(sky_condition.e, blue_bg.e, empty_vec3list.e)
    )
    if depth <= 1:
        return None, None, result_bg

    # Material scatter calculations
    materials: Dict[int, Material] = world.get_materials()
    scattered_list = RayList.new_zero(length)
    attenuation_list = Vec3List.new_zero(length)
    for mat_idx in materials:
        mat_condition = (rec_list.material == mat_idx)
        mat_condition_3 = Vec3List.from_array(mat_condition)
        if not mat_condition.any():
            continue

        ray = RayList(
            Vec3List(cp.where(mat_condition_3.e, r.orig.e, empty_vec3list.e)),
            Vec3List(cp.where(mat_condition_3.e, r.dir.e, empty_vec3list.e))
        )
        rec = HitRecordList(
            Vec3List(cp.where(
                mat_condition_3.e, rec_list.p.e, empty_vec3list.e
            )),
            cp.where(mat_condition, rec_list.t, empty_array_float),
            cp.where(mat_condition, rec_list.material, empty_array_int),
            Vec3List(cp.where(
                mat_condition_3.e, rec_list.normal.e, empty_vec3list.e
            )),
            cp.where(mat_condition, rec_list.front_face, empty_array_bool)
        )
        ray, rec, idx_list = compress(ray, rec)

        scattered, attenuation = materials[mat_idx].scatter(ray, rec)
        scattered, attenuation = decompress(
            scattered, attenuation, idx_list, length
        )
        scattered_list += scattered
        attenuation_list += attenuation

    return scattered_list, attenuation_list, result_bg


def ray_color_loop(r: RayList, world: HittableList, depth: int) -> Vec3List:
    attenuation_list: Dict[int, Vec3List] = dict()
    result_bg_list: Dict[int, Vec3List] = dict()
    length: int = 0
    ray: RayList = r
    for d in range(depth, 0, -1):
        print(f"PID {os.getpid()}: {depth - d} / {depth}   ", end="\r")
        scattered, attenuation, result_bg = ray_color(ray, world, d)
        result_bg_list[length] = result_bg.as_float32()
        if scattered is None or attenuation is None:
            break
        attenuation_list[length] = attenuation.as_float32()
        ray = scattered
        length += 1

    result = result_bg_list[length]
    for i in range(length - 1, -1, -1):
        result = result * attenuation_list[i] + result_bg_list[i]
    return result


def scan_frame(world: HittableList, cam: Camera,
               image_width: int, image_height: int,
               max_depth: int) -> Vec3List:
    length = image_width * image_height
    i_list = cp.tile(cp.arange(image_width), image_height)
    j_list = cp.concatenate(cp.transpose(
        cp.tile(cp.arange(image_height), (image_width, 1))
    ))
    u: cp.ndarray = (random_float_list(length) + i_list) / (image_width - 1)
    v: cp.ndarray = (random_float_list(length) + j_list) / (image_height - 1)
    r: RayList = cam.get_ray(u, v)
    return ray_color_loop(r, world, max_depth)


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

    img_list: List[Vec3List] = Parallel(n_jobs=1, verbose=20)(
        delayed(scan_frame)(
            world, cam, image_width, image_height, max_depth
        ) for s in range(samples_per_pixel)
    )

    # # Profile prologue
    # import cProfile
    # import pstats
    # import io
    # from pstats import SortKey
    # pr = cProfile.Profile()
    # pr.enable()

    # img_list: List[Vec3List] = list()
    # for sample_num in range(samples_per_pixel):
    #     img_list.append(
    #         scan_frame(world, cam, image_width, image_height, max_depth)
    #     )

    # # Profile epilogue
    # pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())

    end_time = time.time()
    print(f"\nDone. Total time: {round(end_time - start_time, 1)} s.")

    final_img = Img(image_width, image_height)
    for img in img_list:
        final_img.write_frame(img)
    final_img.average(samples_per_pixel).gamma(2).up_side_down()
    final_img.save("./output.png", True)


if __name__ == "__main__":
    main()
