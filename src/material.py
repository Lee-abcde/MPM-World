import numpy as np

import taichi as ti
import taichi.math as tm

@ti.func
def Curl_cal(pos, S_grid_v):
    Vz_up = S_grid_v[[pos[0], pos[1] + 1, pos[2]]][2]
    Vz_down = S_grid_v[[pos[0], pos[1] - 1, pos[2]]][2]

    Vy_front = S_grid_v[[pos[0], pos[1], pos[2] + 1]][1]
    Vy_back = S_grid_v[[pos[0], pos[1], pos[2] - 1]][1]

    Vx_front = S_grid_v[[pos[0], pos[1], pos[2] + 1]][0]
    Vx_back = S_grid_v[[pos[0], pos[1], pos[2] - 1]][0]

    Vz_right = S_grid_v[[pos[0] + 1, pos[1], pos[2]]][2]
    Vz_left = S_grid_v[[pos[0] - 1, pos[1], pos[2]]][2]

    Vy_right = S_grid_v[[pos[0] + 1, pos[1], pos[2]]][1]
    Vy_left = S_grid_v[[pos[0] - 1, pos[1], pos[2]]][1]

    Vx_up = S_grid_v[[pos[0], pos[1] + 1, pos[2]]][0]
    Vx_down = S_grid_v[[pos[0], pos[1] - 1, pos[2]]][0]

    dx = 1.
    curl = ti.Vector(
        [(Vz_up - Vz_down) - (Vy_front - Vy_back),
         (Vx_front - Vx_back) - (Vz_right - Vz_left),
         (Vy_right - Vy_left) - (Vx_up - Vx_down)]
    ) / (2. * dx)
    return curl

@ti.func
def CurlGrad_cal(pos, S_grid_c):
    S_right = tm.length(S_grid_c[[pos[0] + 1, pos[1], pos[2]]])
    S_left = tm.length(S_grid_c[[pos[0] - 1, pos[1], pos[2]]])

    S_up = tm.length(S_grid_c[[pos[0], pos[1] + 1, pos[2]]])
    S_down = tm.length(S_grid_c[[pos[0], pos[1] - 1, pos[2]]])

    S_front = tm.length(S_grid_c[[pos[0], pos[1], pos[2] + 1]])
    S_back = tm.length(S_grid_c[[pos[0], pos[1], pos[2] - 1]])

    dx = 1.
    Grad = ti.Vector([S_right - S_left, S_up - S_down, S_front - S_back]) / (2.0 * dx)
    return Grad

@ti.func
def Vorticity_cal(pos, S_grid_c):
    dx = 1.0
    vorticitymult = 5.0

    curl = S_grid_c[pos]
    gradcurl = CurlGrad_cal(pos, S_grid_c)
    vc_force = ti.Vector([0., 0., 0.])
    gradcurllenth = tm.length(gradcurl)
    if gradcurllenth > 1e-5:
        vc_force = vorticitymult * dx * tm.cross(gradcurl / gradcurllenth, curl)

    return vc_force
