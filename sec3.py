from utils.vec3 import Vec3
from utils.img import Img


if __name__ == "__main__":
    w = 2000
    h = 1000
    img = Img(w, h)
    for j in range(h):
        print(f"\rScanlines remaining: {h - j - 1}", end="")
        for i in range(w):
            v = Vec3(i / w, j / h, 0.2)
            img.write_pixel(h - j - 1, i, v)
    print("\nDone.")
    img.save("test.png", True)
