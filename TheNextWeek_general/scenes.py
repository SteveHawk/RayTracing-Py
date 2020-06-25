from typing import Tuple
from utils.vec3 import Vec3, Point3, Color
from utils.sphere import Sphere
from utils.moving_sphere import MovingSphere
from utils.hittable_list import HittableList
from utils.camera import Camera
from utils.material import Lambertian, Metal, Dielectric, DiffuseLight
from utils.rtweekend import random_float
from utils.bvh import BVHNode
from utils.texture import SolidColor, CheckerTexture, NoiseTexture, ImageTexture
from utils.aarect import XYRect, XZRect, YZRect
from utils.hittable import Hittable, FlipFace, RotateY, Translate
from utils.box import Box
from utils.constant_medium import ConstantMedium


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

    pertext = NoiseTexture(4)
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


def earth(aspect_ratio: float, time0: float, time1: float) \
        -> Tuple[BVHNode, Camera]:
    world = HittableList()

    earth_texture = ImageTexture("earthmap.jpg")
    earth_surface = Lambertian(earth_texture)
    globe = Sphere(Point3(0, 0, 0), 2, earth_surface)

    world.add(globe)
    world_bvh = BVHNode(world.objects, time0, time1)

    lookfrom = Point3(0, 0, -5)
    lookat = Point3(0, 0, 0)
    vup = Vec3(0, 1, 0)
    vfov = 50
    dist_to_focus: float = 10
    aperture: float = 0
    cam = Camera(
        lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus,
        time0, time1
    )

    return world_bvh, cam


def simple_light(aspect_ratio: float, time0: float, time1: float) \
        -> Tuple[BVHNode, Camera]:
    world = HittableList()

    pertext = NoiseTexture(4)
    world.add(Sphere(Point3(0, -1000, 0), 1000, Lambertian(pertext)))
    world.add(Sphere(Point3(0, 2, 0), 2, Lambertian(pertext)))

    difflight = DiffuseLight(SolidColor(4, 4, 4))
    world.add(Sphere(Point3(0, 7, 0), 2, difflight))
    world.add(XYRect(3, 5, 1, 3, -2, difflight))

    world_bvh = BVHNode(world.objects, time0, time1)

    lookfrom = Point3(23, 4, 5)
    lookat = Point3(0, 2, 0)
    vup = Vec3(0, 1, 0)
    vfov = 20
    dist_to_focus: float = 10
    aperture: float = 0
    cam = Camera(
        lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus,
        time0, time1
    )

    return world_bvh, cam


def cornell_box(aspect_ratio: float, time0: float, time1: float) \
        -> Tuple[BVHNode, Camera]:
    world = HittableList()

    # Colors
    red = Lambertian(SolidColor(0.65, 0.05, 0.05))
    white = Lambertian(SolidColor(0.73, 0.73, 0.73))
    green = Lambertian(SolidColor(0.12, 0.45, 0.15))
    light = DiffuseLight(SolidColor(15, 15, 15))

    # Outer box
    world.add(FlipFace(YZRect(0, 555, 0, 555, 555, green)))
    world.add(YZRect(0, 555, 0, 555, 0, red))
    world.add(XZRect(213, 343, 227, 332, 554, light))
    world.add(XZRect(0, 555, 0, 555, 0, white))
    world.add(FlipFace(XZRect(0, 555, 0, 555, 555, white)))
    world.add(FlipFace(XYRect(0, 555, 0, 555, 555, white)))

    # Objects in the box
    box1: Hittable = Box(Vec3(0, 0, 0), Point3(165, 330, 165), white)
    box1 = RotateY(box1, 15)
    box1 = Translate(box1, Point3(265, 0, 295))
    box1 = ConstantMedium(box1, 0.01, SolidColor(0, 0, 0))
    world.add(box1)

    box2: Hittable = Box(Point3(0, 0, 0), Point3(165, 165, 165), white)
    box2 = RotateY(box2, -18)
    box2 = Translate(box2, Point3(130, 0, 65))
    box2 = ConstantMedium(box2, 0.01, SolidColor(1, 1, 1))
    world.add(box2)

    world_bvh = BVHNode(world.objects, time0, time1)

    lookfrom = Point3(278, 278, -800)
    lookat = Point3(278, 278, 0)
    vup = Vec3(0, 1, 0)
    vfov = 40
    dist_to_focus: float = 10
    aperture: float = 0
    cam = Camera(
        lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus,
        time0, time1
    )

    return world_bvh, cam


def final_scene(aspect_ratio: float, time0: float, time1: float) \
        -> Tuple[BVHNode, Camera]:
    world = HittableList()

    # Ground
    boxes1 = HittableList()
    ground = Lambertian(SolidColor(0.48, 0.83, 0.53))
    boxes_per_side = 20
    for i in range(boxes_per_side):
        for j in range(boxes_per_side):
            w = 100
            x0 = -1000 + i*w
            z0 = -1000 + j*w
            y0 = 0
            x1 = x0 + w
            y1 = random_float(1, 101)
            z1 = z0 + w
            world.add(Box(Point3(x0, y0, x0), Point3(x1, y1, z1), ground))
    world.add(BVHNode(boxes1.objects, time0, time1))

    # Light
    light = DiffuseLight(SolidColor(7, 7, 7))
    world.add(XZRect(123, 423, 147, 412, 554, light))

    # Moving sphere
    center1 = Point3(400, 400, 200)
    center2 = center1 + Vec3(30, 0, 0)
    moving_sphere_material = Lambertian(SolidColor(0.7, 0.3, 0.1))
    world.add(MovingSphere(center1, center2, 0, 1, 50, moving_sphere_material))

    # Dielectric & metal balls
    world.add(Sphere(Point3(260, 150, 45), 50, Dielectric(1.5)))
    world.add(
        Sphere(Point3(0, 150, 145), 50, Metal(Color(0.8, 0.8, 0.9), 10.0))
    )

    # The subsurface reflection sphere
    boundary = Sphere(Point3(360, 150, 145), 70, Dielectric(1.5))
    world.add(boundary)
    world.add(ConstantMedium(boundary, 0.2, SolidColor(0.2, 0.4, 0.9)))

    # Big thin mist
    mist = Sphere(Point3(0, 0, 0), 5000, Dielectric(1.5))
    world.add(ConstantMedium(mist, 0.0001, SolidColor(1, 1, 1)))

    # Earth and marble ball
    emat = Lambertian(ImageTexture("earthmap.jpg"))
    world.add(Sphere(Point3(400, 200, 400), 100, emat))
    pertext = NoiseTexture(0.1)
    world.add(Sphere(Point3(220, 280, 300), 80, Lambertian(pertext)))

    # Foam
    boxes2 = HittableList()
    white = Lambertian(SolidColor(0.73, 0.73, 0.73))
    ns = 1000
    for j in range(ns):
        boxes2.add(Sphere(Point3.random(0, 165), 10, white))
    world.add(Translate(
        RotateY(
            BVHNode(boxes2.objects, time0, time1), 15
        ), Vec3(-100, 270, 395)
    ))

    world_bvh = BVHNode(world.objects, time0, time1)

    lookfrom = Point3(478, 278, -600)
    lookat = Point3(278, 278, 0)
    vup = Vec3(0, 1, 0)
    vfov = 40
    dist_to_focus: float = 10
    aperture: float = 0
    cam = Camera(
        lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus,
        time0, time1
    )

    return world_bvh, cam
