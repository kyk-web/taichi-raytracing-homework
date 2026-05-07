import taichi as ti
from taichi import math as tm

from .config import (
    EPS,
    MAX_BOUNCES_LIMIT,
    BACKGROUND_COLOR,
    MIRROR_REFLECTIVITY,
    MAT_MIRROR,
    MAT_RED,
)
from .geometry import intersect_scene
from .math_utils import safe_normalize, reflect, clamp01


@ti.func
def calculate_local_light(pos, normal, view_dir, base_color, material_id, light_pos):
    ambient_strength = 0.08
    light_color = tm.vec3(1.0, 1.0, 1.0)

    color = ambient_strength * base_color

    to_light = light_pos - pos
    light_distance = tm.length(to_light)
    light_dir = safe_normalize(to_light)

    shadow_origin = pos + normal * EPS

    shadow_hit, shadow_t, _, _, _, _ = intersect_scene(shadow_origin, light_dir)

    in_shadow = 0
    if shadow_hit == 1 and shadow_t < light_distance - EPS:
        in_shadow = 1

    if in_shadow == 0:
        ndotl = ti.max(tm.dot(normal, light_dir), 0.0)

        diffuse = ndotl * base_color * light_color

        specular = tm.vec3(0.0, 0.0, 0.0)

        if material_id == MAT_RED:
            reflect_dir = reflect(-light_dir, normal)
            spec = ti.pow(ti.max(tm.dot(view_dir, reflect_dir), 0.0), 48.0)
            specular = 0.15 * spec * light_color

        color += diffuse + specular

    return clamp01(color)


@ti.func
def trace_ray(ray_origin, ray_dir, light_pos, max_bounces):
    final_color = tm.vec3(0.0, 0.0, 0.0)
    throughput = tm.vec3(1.0, 1.0, 1.0)

    current_origin = ray_origin
    current_dir = ray_dir

    for bounce in range(MAX_BOUNCES_LIMIT):
        if bounce >= max_bounces:
            break

        hit, hit_t, hit_pos, hit_normal, hit_color, hit_material = intersect_scene(
            current_origin,
            current_dir,
        )

        if hit == 0:
            final_color += throughput * tm.vec3(
                BACKGROUND_COLOR[0],
                BACKGROUND_COLOR[1],
                BACKGROUND_COLOR[2],
            )
            break

        if hit_material == MAT_MIRROR:
            reflected_dir = safe_normalize(reflect(current_dir, hit_normal))

            current_origin = hit_pos + hit_normal * EPS
            current_dir = reflected_dir

            throughput *= MIRROR_REFLECTIVITY
        else:
            view_dir = safe_normalize(-current_dir)

            local_color = calculate_local_light(
                hit_pos,
                hit_normal,
                view_dir,
                hit_color,
                hit_material,
                light_pos,
            )

            final_color += throughput * local_color
            break

    return clamp01(final_color)
