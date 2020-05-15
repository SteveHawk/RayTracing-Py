from utils.vec3 import Vec3, Point3
from utils.ray import Ray
from utils.hittable import hittable, hit_record


class sphere(hittable):
    def __init__(self, center: Point3 = Point3(), r: float = 0):
        self.center = center
        self.radius = r
    
    @classmethod
    def hit(cls, r: Ray, t_min: float, t_max: float, rec: hit_record) -> bool:
        return True
