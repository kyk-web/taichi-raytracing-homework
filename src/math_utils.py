import taichi as ti
from taichi import math as tm


@ti.func
def safe_normalize(v):
    length = tm.length(v)
    result = tm.vec3(0.0, 0.0, 0.0)
    if length > 1e-8:
        result = v / length
    return result


@ti.func
def reflect(v, n):
    return v - 2.0 * tm.dot(v, n) * n


@ti.func
def clamp01(c):
    return tm.clamp(c, 0.0, 1.0)
