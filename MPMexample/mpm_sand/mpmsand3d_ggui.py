# 这是一个从耦合代码中复制过来，希望先调通3d沙子模拟的产物
# written by @Lee-abcde
import numpy as np

import taichi as ti

import src.material as material
import src.grid as grid
arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)

#dim, n_grid, steps, dt = 2, 128, 20, 2e-4
#dim, n_grid, steps, dt = 2, 256, 32, 1e-4
# dim, n_grid, steps, dt = 3, 32, 25, 4e-4
dim, n_grid, steps, dt = 3, 64, 25, 2e-4
# dim, n_grid, steps, dt = 3, 128, 5, 1e-4
#######################
# Simulation parameters
#######################
n_particles = n_grid**dim // 2**(dim - 1)

print(n_particles)

dx, inv_dx = 1 / n_grid, float(n_grid)

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
GRAVITY = [0, -9.8, 0]
bound = 3
E = 1000  # Young's modulus
nu = 0.2  #  Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters

#   Sand parameters
s_rho = 400
s_mass = p_vol * s_rho
E_s, nu_s = 3.537e5, 0.3 # sand's Young's modulus and Poisson's ratio
mu_s, lambda_s = E_s / (2 * (1 + nu_s)), E_s * nu_s / ((1 + nu_s) * (1 - 2 * nu_s)) # sand's Lame parameters

mu_b = 0.75  # coefficient of friction

pi = 3.14159265358979
#######################
# Particle data structure
#######################
F_x = ti.Vector.field(dim, float, n_particles)
F_v = ti.Vector.field(dim, float, n_particles)
F_C = ti.Matrix.field(dim, dim, float, n_particles)
F_dg = ti.Matrix.field(3, 3, dtype=float,
                       shape=n_particles)  # deformation gradient
F_Jp = ti.field(float, n_particles)

F_colors = ti.Vector.field(4, float, n_particles)
F_materials = ti.field(int, n_particles)

alpha_s = ti.field(dtype = float, shape = n_particles)  # sand yield surface size
q_s = ti.field(dtype = float, shape = n_particles) # harding state
#######################
# Grid data structure
#######################
F_grid_v = ti.Vector.field(dim, float, (n_grid, ) * dim)
F_grid_m = ti.field(float, (n_grid, ) * dim)
S_grid_v = ti.Vector.field(dim, float, (n_grid, ) * dim)
S_grid_m = ti.field(float, (n_grid, ) * dim)
S_grid_c = ti.Vector.field(dim, float, (n_grid, ) * dim) # field storage curl

neighbour = (3, ) * dim

WATER = 0
JELLY = 1
SNOW = 2
SMOKE = 3
SAND = 4


@ti.func
def sand_project(e0, p):
    e = e0
    ehat = e - e.trace() / dim * ti.Matrix.identity(float, dim)  # 公式（27）
    Fnorm = ti.sqrt(ehat[0, 0] ** 2 + ehat[1, 1] ** 2 + ehat[2, 2] ** 2)  # Frobenius norm
    yp = Fnorm + (dim * lambda_s + 2 * mu_s) / (2 * mu_s) * e.trace() * alpha_s[p]  # delta gamma 公式（27）

    new_e = ti.Matrix.zero(float, dim, dim)
    delta_q = 0.0
    if Fnorm <= 0 or e.trace() > 0:  # Case II:
        new_e = ti.Matrix.zero(float, dim, dim)
        delta_q = ti.sqrt(e[0, 0] ** 2 + e[1, 1] ** 2 + e[2, 2] ** 2)
    elif yp <= 0:  # Case I:
        new_e = e0  # return initial matrix without volume correction and cohesive effect
        delta_q = 0
    else:  # Case III:
        new_e = e - yp / Fnorm * ehat
        delta_q = yp

    return new_e, delta_q

h0, h1, h2, h3 = 35, 9, 0.2, 10
@ti.func
def sand_hardening(dq, p): # The amount of hardening depends on the amount of correction that occurred due to plasticity
    q_s[p] += dq  # 公式（29）
    phi = h0 + (h1 * q_s[p] - h3) * ti.exp(-h2 * q_s[p])  # 公式（30）
    phi = phi / 180 * pi  # details in Table. 3: Friction angle phi_F and hardening parameters h0, h1, and h3 are listed in degrees for convenience
    sin_phi = ti.sin(phi)
    alpha_s[p] = ti.sqrt(2 / 3) * (2 * sin_phi) / (3 - sin_phi)  # 公式（31）

@ti.kernel
def substep(g_x: float, g_y: float, g_z: float):
    for I in ti.grouped(F_grid_m):
        F_grid_v[I] = ti.zero(F_grid_v[I])
        F_grid_m[I] = 0

    # Step1: Particle to grid (P2G)
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        F_dg[p] = (ti.Matrix.identity(float, 3) + dt * F_C[p]) @ F_dg[p]  # deformation gradient update

        U, sig, V = ti.svd(F_dg[p])
        e = ti.Matrix([[ti.log(sig[0, 0]), 0, 0], [0, ti.log(sig[1, 1]), 0],
                       [0, 0, ti.log(sig[2, 2])]])  # Epsilon定义在论文公式(27)上方
        new_e, dq = sand_project(e, p)  # dq指代论文中的δqp，公式（29）上方有定义
        sand_hardening(dq, p)
        new_F = U @ ti.Matrix(
            [[ti.exp(new_e[0, 0]), 0, 0], [0, ti.exp(new_e[1, 1]), 0], [0, 0, ti.exp(new_e[2, 2])]]) @ V.transpose()
        F_dg[p] = new_F
        e = new_e

        U, sig, V = ti.svd(F_dg[p])
        inv_sig = sig.inverse()
        pd_psi_F = U @ (2 * mu_s * inv_sig @ e + lambda_s * e.trace() * inv_sig) @ V.transpose()  # 公式 (26)
        stress = (-p_vol * 4 * inv_dx * inv_dx) * pd_psi_F @ F_dg[p].transpose()  # 公式（23）
        affine = dt * stress + s_mass * F_C[p]

        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            F_grid_v[base +
                     offset] += weight * (s_mass * F_v[p] + affine @ dpos)
            F_grid_m[base + offset] += weight * s_mass

    # Step2: Grid operations
    for I in ti.grouped(F_grid_m):
        if F_grid_m[I] > 0:
            F_grid_v[I] /= F_grid_m[I]
        F_grid_v[I] += dt * ti.Vector([g_x, g_y, g_z])
        grid.boundary_separate(I, F_grid_v, n_grid, bound)

    # Step3: Grid to particle (G2P)
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, dim)
        new_C = ti.Matrix.zero(float, dim, dim)
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = F_grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        F_v[p], F_C[p] = new_v, new_C
        F_x[p] += dt * F_v[p]

@ti.kernel
def set_color_by_material(mat_color: ti.types.ndarray()):
    for i in range(n_particles):
        mat = F_materials[i]
        F_colors[i] = ti.Vector(
            [mat_color[mat, 0], mat_color[mat, 1], mat_color[mat, 2], 1.0])



particles_radius = 0.003

material_colors = [(0.1, 0.6, 0.9), (0.93, 0.33, 0.23), (1.0, 1.0, 1.0), (0.5, 0.5, 0.5), (0.4, 0.3, 0.2)]



@ti.kernel
def init_sand():
    group_size = n_particles // 4
    for i in range(n_particles):
        F_x[i] = [
            ti.random() * 0.2 + 0.7 - 0.2 * (i // group_size),
            ti.random() * 0.2 + 0.7 - 0.2 * (i // group_size),
            ti.random() * 0.2 + 0.7 - 0.2 * (i // group_size)
        ]
        F_v[i] = ti.Vector([0, 0, 0])
        F_C[i] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        F_dg[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        F_Jp[i] = 1
        alpha_s[i] = 0.267765
        F_materials[i] = 4  # 0: fluid 1: jelly 2: snow 3:smoke 4:sand

    # for i in F_x:
    #     F_x[i] = [ti.random() * 0.25 + 0.4, ti.random() * 0.4 + 0.2,ti.random() * 0.25 + 0.2]
    #     F_v[i] = ti.Matrix([0, 0, 0])
    #     F_dg[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    #     alpha_s[i] = 0.267765
    #     F_materials[i] = 4  # 0: fluid 1: jelly 2: snow 3:smoke 4:sand



res = (1080, 720)
window = ti.ui.Window("Real MPM 3D", res, vsync=True)

canvas = window.get_canvas()
gui = window.get_gui()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.3, 0.5)
camera.fov(55)


def show_options():
        set_color_by_material(np.array(material_colors, dtype=np.float32))
def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))

    colors_used = F_colors
    scene.particles(F_x, per_vertex_color=colors_used, radius=particles_radius)

    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

    canvas.scene(scene)


def main():
    init_sand()
    show_options()
    while window.running:
        for _ in range(steps):
            substep(*GRAVITY)
        render()
        window.show()


if __name__ == '__main__':
    main()