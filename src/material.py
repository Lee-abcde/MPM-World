import taichi as ti
import taichi.math as tm

@ti.func
def Curl_cal(pos, S_grid_v):
    vz_up = S_grid_v[[pos[0], pos[1] + 1, pos[2]]][2]
    vz_down = S_grid_v[[pos[0], pos[1] - 1, pos[2]]][2]

    vy_front = S_grid_v[[pos[0], pos[1], pos[2] + 1]][1]
    vy_back = S_grid_v[[pos[0], pos[1], pos[2] - 1]][1]

    vx_front = S_grid_v[[pos[0], pos[1], pos[2] + 1]][0]
    vx_back = S_grid_v[[pos[0], pos[1], pos[2] - 1]][0]

    vz_right = S_grid_v[[pos[0] + 1, pos[1], pos[2]]][2]
    vz_left = S_grid_v[[pos[0] - 1, pos[1], pos[2]]][2]

    vy_right = S_grid_v[[pos[0] + 1, pos[1], pos[2]]][1]
    vy_left = S_grid_v[[pos[0] - 1, pos[1], pos[2]]][1]

    vx_up = S_grid_v[[pos[0], pos[1] + 1, pos[2]]][0]
    vx_down = S_grid_v[[pos[0], pos[1] - 1, pos[2]]][0]

    dx = 1.
    curl = ti.Vector(
        [(vz_up - vz_down) - (vy_front - vy_back),
         (vx_front - vx_back) - (vz_right - vz_left),
         (vy_right - vy_left) - (vx_up - vx_down)]
    ) / (2. * dx)
    return curl

@ti.func
def Curl_cal2D(pos, S_grid_v):
    vy_right = S_grid_v[[pos[0] + 1, pos[1]]][1]
    vy_left = S_grid_v[[pos[0] - 1, pos[1]]][1]

    vx_up = S_grid_v[[pos[0], pos[1] + 1]][0]
    vx_down = S_grid_v[[pos[0], pos[1] - 1]][0]

    dx = 1.
    curl = (vy_right - vy_left - vx_up + vx_down) / (2. * dx)
    return curl
@ti.func
def CurlGrad_cal(pos, S_grid_c):
    s_right = tm.length(S_grid_c[[pos[0] + 1, pos[1], pos[2]]])
    s_left = tm.length(S_grid_c[[pos[0] - 1, pos[1], pos[2]]])

    s_up = tm.length(S_grid_c[[pos[0], pos[1] + 1, pos[2]]])
    s_down = tm.length(S_grid_c[[pos[0], pos[1] - 1, pos[2]]])

    s_front = tm.length(S_grid_c[[pos[0], pos[1], pos[2] + 1]])
    s_back = tm.length(S_grid_c[[pos[0], pos[1], pos[2] - 1]])

    dx = 1.
    curlgrad = ti.Vector([s_right - s_left, s_up - s_down, s_front - s_back]) / (2.0 * dx)
    return curlgrad

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

@ti.func
def Vorticity_cal2D(pos, S_grid_c):
    dx = 1.0
    vorticitymult = 1.0

    curl = ti.Vector([0., 0., S_grid_c[pos]])
    s_right = S_grid_c[[pos[0] + 1, pos[1]]]
    s_left = S_grid_c[[pos[0] - 1, pos[1]]]

    s_up = S_grid_c[[pos[0], pos[1] + 1]]
    s_down = S_grid_c[[pos[0], pos[1] - 1]]
    gradcurl = ti.Vector([s_right - s_left, s_up - s_down, 0]) / (2.0 * dx)
    vc_force = ti.Vector([0., 0., 0])
    gradcurllenth = tm.length(gradcurl)
    if gradcurllenth > 1e-5:
        vc_force = vorticitymult * dx * tm.cross(gradcurl / gradcurllenth, curl)

    return ti.Vector([vc_force[0],vc_force[1]])
