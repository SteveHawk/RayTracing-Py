from typing import Tuple
from utils.vec3 import Vec3, Point3, Color
from utils.sphere import Sphere
from utils.moving_sphere import MovingSphere
from utils.hittable_list import HittableList
from utils.camera import Camera
from utils.material import Lambertian, Metal, Dielectric
from utils.rtweekend import random_float
from utils.bvh import BVHNode
from utils.texture import SolidColor, CheckerTexture, NoiseTexture


def three_ball_scene(aspect_ratio: float, time0: float, time1: float) \
        -> Tuple[BVHNode, Camera]:
    world = HittableList()
    world.add(Sphere(
        Point3(0, 0, -1), 0.5, Lambertian(SolidColor(0.1, 0.2, 0.5))
    ))
    world.add(Sphere(
        Point3(0, -100.5, -1), 100, Lambertian(SolidColor(0.8, 0.8, 0))
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
    world_bvh = BVHNode(world.objects, time0, time1)

    lookfrom = Point3(3, 3, 2)
    lookat = Point3(0, 0, -1)
    vup = Vec3(0, 1, 0)
    vfov = 20
    dist_to_focus: float = (lookfrom - lookat).length()
    aperture: float = 0
    cam = Camera(
        lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus,
        time0, time1
    )

    # lookfrom = Point3(13, 2, 3)
    # lookat = Point3(0, 0, 0)
    # vup = Vec3(0, 1, 0)
    # vfov = 20
    # dist_to_focus: float = 10
    # aperture: float = 0.1
    # cam = Camera(
    #     lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus,
    #     time0, time1
    # )

    return world_bvh, cam


def random_scene(aspect_ratio: float, time0: float, time1: float) \
        -> Tuple[BVHNode, Camera]:
    world = HittableList()

    # ground_material = Lambertian(SolidColor(0.5, 0.5, 0.5))
    ground_material = Lambertian(CheckerTexture(
        SolidColor(0.2, 0.3, 0.1),
        SolidColor(0.9, 0.9, 0.9)
    ))
    world.add(Sphere(Point3(0, -1000, 0), 1000, ground_material))

    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = random_float()
            center = Point3(
                a + 0.9*random_float(), 0.2, b + 0.9*random_float()
            )

            if (center - Vec3(4, 0.2, 0)).length() > 0.9:
                if choose_mat < 0.6:
                    # Diffuse
                    albedo = Color.random() * Color.random()
                    sphere_material_diffuse = Lambertian(SolidColor(albedo))
                    center2 = center + Vec3(0, random_float(0, 0.5), 0)
                    world.add(MovingSphere(
                        center, center2, 0, 1, 0.2, sphere_material_diffuse
                    ))
                elif choose_mat < 0.8:
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

    material_2 = Lambertian(SolidColor(0.4, 0.2, 0.1))
    world.add(Sphere(Point3(-4, 1, 0), 1, material_2))

    material_3 = Metal(Color(0.7, 0.6, 0.5), 0)
    world.add(Sphere(Point3(4, 1, 0), 1, material_3))

    world_bvh = BVHNode(world.objects, time0, time1)

    lookfrom = Point3(13, 2, 3)
    lookat = Point3(0, 0, 0)
    vup = Vec3(0, 1, 0)
    vfov = 20
    dist_to_focus: float = 10
    aperture: float = 0.1
    cam = Camera(
        lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus,
        time0, time1
    )

    return world_bvh, cam


def two_spheres(aspect_ratio: float, time0: float, time1: float) \
        -> Tuple[BVHNode, Camera]:
    world = HittableList()

    checker = CheckerTexture(
        SolidColor(0.2, 0.3, 0.1),
        SolidColor(0.9, 0.9, 0.9)
    )
    world.add(Sphere(Point3(0, -10, 0), 10, Lambertian(checker)))
    world.add(Sphere(Point3(0, 10, 0), 10, Lambertian(checker)))

    world_bvh = BVHNode(world.objects, time0, time1)

    lookfrom = Point3(13, 2, 3)
    lookat = Point3(0, 0, 0)
    vup = Vec3(0, 1, 0)
    vfov = 20
    dist_to_focus: float = 10
    aperture: float = 0
    cam = Camera(
        lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus,
        time0, time1
    )

    return world_bvh, cam


def two_perlin_spheres(aspect_ratio: float, time0: float, time1: float) \
        -> Tuple[BVHNode, Camera]:
    world = HittableList()

    pertext = NoiseTexture()
    world.add(Sphere(Point3(0, -1000, 0), 1000, Lambertian(pertext)))
    world.add(Sphere(Point3(0, 2, 0), 2, Lambertian(pertext)))

    world_bvh = BVHNode(world.objects, time0, time1)

    lookfrom = Point3(13, 2, 3)
    lookat = Point3(0, 0, 0)
    vup = Vec3(0, 1, 0)
    vfov = 20
    dist_to_focus: float = 10
    aperture: float = 0
    cam = Camera(
        lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus,
        time0, time1
    )

    return world_bvh, cam
