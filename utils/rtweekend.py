import numpy as np  # type: ignore
import random
random.seed()


# Utility Functions
def degrees_to_radians(degrees: float) -> float:
    return degrees * np.pi / 180


def random_float(_min: float = None, _max: float = None) -> float:
    if _min is None and _max is None:
        return random.random()
    elif isinstance(_min, (int, float)) and isinstance(_max, (int, float)):
        return random.uniform(_min, _max)
    else:
        raise TypeError


def clamp(x: float, _min: float, _max: float) -> float:
    if x < _min:
        return _min
    elif x > _max:
        return _max
    return x
