import taichi as ti
from taichi import math as tm

from .config import (
    EPS,
    INF,
    GROUND_Y,
    CHECKER_SCALE,
    CHECKER_DARK,
    CHECKER_LIGHT,
    RED_SPHERE_CENTER,
    MIRROR_SPHERE_CENTER,
    SPHERE_RADIUS,
    RED_COLOR,
    MAT_GROUND,
    MAT_RED,
    MAT_MIRROR,
)
from .math_utils import safe_normalize


@ti.func
def checker_color(pos):
    pattern = ti.cast(
        ti.floor(pos.x * CHECKER_SCALE) + ti.floor(pos.z * CHECKER_SCALE),
        ti.i32,
    ) % 2

    color = tm.vec3(CHECKER_LIGHT[0], CHECKER_LIGHT[1], CHECKER_LIGHT[2])
    if pattern == 0:
        color = tm.vec3(CHECKER_DARK[0], CHECKER_DARK[1], CHECKER_DARK[2])

    return color


@ti.func
def intersect_sphere(ray_origin, ray_dir, center, radius):
    hit = 0
    hit_t = INF
    hit_normal = tm.vec3(0.0, 0.0, 0.0)

    oc = ray_origin - center

    b = tm.dot(oc, ray_dir)
    c = tm.dot(oc, oc) - radius * radius
    discriminant = b * b - c

    if discriminant > 0.0:
        sqrt_d = ti.sqrt(discriminant)

        t0 = -b - sqrt_d
        t1 = -b + sqrt_d

        t = t0
        if t < EPS:
            t = t1

        if t > EPS:
            hit = 1
            hit_t = t

            hit_pos = ray_origin + ray_dir * hit_t
            hit_normal = safe_normalize(hit_pos - center)

    return hit, hit_t, hit_normal


@ti.func
def intersect_plane(ray_origin, ray_dir):
    hit = 0
    hit_t = INF
    hit_normal = tm.vec3(0.0, 1.0, 0.0)

    if ti.abs(ray_dir.y) > 1e-6:
        t = (GROUND_Y - ray_origin.y) / ray_dir.y

        if t > EPS:
            hit = 1
            hit_t = t

    return hit, hit_t, hit_normal


@ti.func
def intersect_scene(ray_origin, ray_dir):
    hit = 0
    closest_t = INF

    hit_pos = tm.vec3(0.0, 0.0, 0.0)
    hit_normal = tm.vec3(0.0, 1.0, 0.0)
    hit_color = tm.vec3(0.0, 0.0, 0.0)
    hit_material = -1

    plane_hit, plane_t, plane_normal = intersect_plane(ray_origin, ray_dir)

    if plane_hit == 1 and plane_t < closest_t:
        hit = 1
        closest_t = plane_t
        hit_pos = ray_origin + ray_dir * closest_t
        hit_normal = plane_normal
        hit_color = checker_color(hit_pos)
        hit_material = MAT_GROUND

    red_center = tm.vec3(
        RED_SPHERE_CENTER[0],
        RED_SPHERE_CENTER[1],
        RED_SPHERE_CENTER[2],
    )

    red_hit, red_t, red_normal = intersect_sphere(
        ray_origin,
        ray_dir,
        red_center,
        SPHERE_RADIUS,
    )

    if red_hit == 1 and red_t < closest_t:
        hit = 1
        closest_t = red_t
        hit_pos = ray_origin + ray_dir * closest_t
        hit_normal = red_normal
        hit_color = tm.vec3(RED_COLOR[0], RED_COLOR[1], RED_COLOR[2])
        hit_material = MAT_RED

    mirror_center = tm.vec3(
        MIRROR_SPHERE_CENTER[0],
        MIRROR_SPHERE_CENTER[1],
        MIRROR_SPHERE_CENTER[2],
    )

    mirror_hit, mirror_t, mirror_normal = intersect_sphere(
        ray_origin,
        ray_dir,
        mirror_center,
        SPHERE_RADIUS,
    )

    if mirror_hit == 1 and mirror_t < closest_t:
        hit = 1
        closest_t = mirror_t
        hit_pos = ray_origin + ray_dir * closest_t
        hit_normal = mirror_normal
        hit_color = tm.vec3(0.85, 0.85, 0.85)
        hit_material = MAT_MIRROR

    return hit, closest_t, hit_pos, hit_normal, hit_color, hit_material
