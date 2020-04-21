from utils.vec3 import Vec3
from utils.img import Img
from utils.ray import Ray


def ray_color(r: Ray) -> Vec3:
    unit_direction: Vec3 = r.direction().unit_vector()
    t: float = (unit_direction.y() + 1) * 0.5
    return Vec3(1, 1, 1) * (1 - t) + Vec3(0.5, 0.7, 1) * t


if __name__ == "__main__":
    image_width = 200
    image_height = 100
    img = Img(image_width, image_height)

    lower_left_corner = Vec3(-2, -1, -1)
    horizontal = Vec3(4, 0, 0)
    vertical = Vec3(0, 2, 0)
    origin = Vec3(0, 0, 0)

    for j in range(image_height - 1, -1, -1):
        print(f"\rScanlines remaining: {j} ", end="")
        for i in range(image_width):
            u = i / image_width
            v = j / image_height
            r = Ray(origin, lower_left_corner + horizontal*u + vertical*v)
            color = ray_color(r)
            img.write_pixel(i, image_height - j - 1, color)
    print("Done.")

    img.save("./test.png", True)
