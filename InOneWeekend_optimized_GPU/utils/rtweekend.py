import cupy as cp  # type: ignore
# rng = cp.random.default_rng()
rng = cp.random


# Utility Functions
def degrees_to_radians(degrees: float) -> float:
    return degrees * cp.pi / 180


def random_float(_min: float = 0, _max: float = 1) -> float:
    return rng.uniform(_min, _max)


def random_float_list(size: int, _min: float = 0, _max: float = 1) \
        -> cp.ndarray:
    return rng.uniform(_min, _max, size).astype(cp.float32, copy=False)
