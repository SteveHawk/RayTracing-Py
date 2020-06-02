import numpy as np  # type: ignore
rng = np.random.default_rng()


# Utility Functions
def degrees_to_radians(degrees: float) -> float:
    return degrees * np.pi / 180


def random_float(_min: float = 0, _max: float = 1) -> float:
    return rng.uniform(_min, _max)


def random_float_list(size: int, _min: float = 0, _max: float = 1) \
        -> np.ndarray:
    return rng.uniform(_min, _max, size)
