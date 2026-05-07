import math

import taichi as ti
from taichi import math as tm

from .config import (
    WIDTH,
    HEIGHT,
    FOV_DEG,
    CAMERA_POS,
    CAMERA_TARGET,
    MAX_BOUNCES_LIMIT,
)
from .math_utils import safe_normalize
from .shading import trace_ray


try:
    ti.init(arch=ti.gpu, default_fp=ti.f32)
except Exception:
    ti.init(arch=ti.cpu, default_fp=ti.f32)


pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))


@ti.kernel
def render(light_x: ti.f32, light_y: ti.f32, light_z: ti.f32, max_bounces: ti.i32):
    camera_pos = tm.vec3(CAMERA_POS[0], CAMERA_POS[1], CAMERA_POS[2])
    camera_target = tm.vec3(CAMERA_TARGET[0], CAMERA_TARGET[1], CAMERA_TARGET[2])

    forward = safe_normalize(camera_target - camera_pos)
    right = safe_normalize(tm.cross(forward, tm.vec3(0.0, 1.0, 0.0)))
    up = safe_normalize(tm.cross(right, forward))

    aspect = WIDTH / HEIGHT
    tan_half_fov = ti.tan(math.radians(FOV_DEG) * 0.5)

    light_pos = tm.vec3(light_x, light_y, light_z)

    for i, j in pixels:
        x = (2.0 * (ti.cast(i, ti.f32) + 0.5) / WIDTH - 1.0) * aspect * tan_half_fov
        y = (2.0 * (ti.cast(j, ti.f32) + 0.5) / HEIGHT - 1.0) * tan_half_fov

        ray_dir = safe_normalize(forward + x * right + y * up)

        color = trace_ray(camera_pos, ray_dir, light_pos, max_bounces)

        pixels[i, j] = color


def run():
    window = ti.ui.Window(
        "Whitted Ray Tracing Demo",
        (WIDTH, HEIGHT),
        vsync=True,
    )
    canvas = window.get_canvas()
    gui = window.get_gui()

    light_x = 2.0
    light_y = 4.0
    light_z = 3.0

    max_bounces = 3

    while window.running:
        render(light_x, light_y, light_z, max_bounces)

        canvas.set_image(pixels)

        with gui.sub_window("Controls", 0.72, 0.05, 0.26, 0.24):
            light_x = gui.slider_float("Light X", light_x, -5.0, 5.0)
            light_y = gui.slider_float("Light Y", light_y, 0.5, 8.0)
            light_z = gui.slider_float("Light Z", light_z, -5.0, 8.0)
            max_bounces = gui.slider_int(
                "Max Bounce",
                max_bounces,
                1,
                MAX_BOUNCES_LIMIT,
            )

        window.show()
