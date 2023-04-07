# written by @Lee-abcde
import numpy as np

import taichi as ti

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)

#dim, n_grid, steps, dt = 2, 128, 20, 2e-4
#dim, n_grid, steps, dt = 2, 256, 32, 1e-4
# dim, n_grid, steps, dt = 3, 32, 25, 4e-4
dim, n_grid, steps, dt = 3, 64, 25, 2e-4
# dim, n_grid, steps, dt = 3, 128, 5, 1e-4

n_particles = n_grid**dim // 2**(dim - 1)

print(n_particles)

dx = 1 / n_grid

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
GRAVITY = [0, -9.8, 0]
bound = 3
E = 1000  # Young's modulus
nu = 0.2  #  Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters

F_x = ti.Vector.field(dim, float, n_particles)
F_v = ti.Vector.field(dim, float, n_particles)
F_C = ti.Matrix.field(dim, dim, float, n_particles)
F_dg = ti.Matrix.field(3, 3, dtype=float,
                       shape=n_particles)  # deformation gradient
F_Jp = ti.field(float, n_particles)

F_colors = ti.Vector.field(4, float, n_particles)
F_colors_random = ti.Vector.field(4, float, n_particles)
F_materials = ti.field(int, n_particles)
F_grid_v = ti.Vector.field(dim, float, (n_grid, ) * dim)
F_grid_m = ti.field(float, (n_grid, ) * dim)

S_grid_v = ti.Vector.field(dim, float, (n_grid, ) * dim)
S_grid_m = ti.field(float, (n_grid, ) * dim)

neighbour = (3, ) * dim

WATER = 0
JELLY = 1
SNOW = 2
SMOKE = 3

@ti.kernel
def substep(g_x: float, g_y: float, g_z: float):
    for I in ti.grouped(F_grid_m):
        F_grid_v[I] = ti.zero(F_grid_v[I])
        F_grid_m[I] = 0
        S_grid_v[I] = ti.zero(F_grid_v[I])
        S_grid_m[I] = 0
    # Step1: Particle to grid (P2G)
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

        F_dg[p] = (ti.Matrix.identity(float, 3) + dt * F_C[p]) @ F_dg[p]  # deformation gradient update
        # Hardening coefficient: snow gets harder when compressed
        h = ti.exp(10 * (1.0 - F_Jp[p]))
        if F_materials[p] == JELLY:  # jelly, make it softer
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if F_materials[p] == WATER or F_materials[p] == SMOKE:  # liquid
            mu = 0.0

        U, sig, V = ti.svd(F_dg[p])
        J = 1.0
        for d in ti.static(range(3)):
            new_sig = sig[d, d]
            if F_materials[p] == SNOW:  # Snow
                new_sig = ti.min(ti.max(sig[d, d], 1 - 2.5e-2),
                                 1 + 4.5e-3)  # Plasticity
            F_Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if F_materials[p] == WATER or F_materials[p] == SMOKE:
            # Reset deformation gradient to avoid numerical instability
            new_F = ti.Matrix.identity(float, 3)
            new_F[0, 0] = J
            F_dg[p] = new_F
        elif F_materials[p] == SNOW:
            # Reconstruct elastic deformation gradient after plasticity
            F_dg[p] = U @ sig @ V.transpose()
        stress = 2 * mu * (F_dg[p] - U @ V.transpose()) @ F_dg[p].transpose(
        ) + ti.Matrix.identity(float, 3) * la * J * (J - 1)
        stress = (-dt * p_vol * 4) * stress / dx**2
        affine = stress + p_mass * F_C[p]

        if F_materials[p] == SMOKE:
            for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
                dpos = (offset - fx) * dx
                weight = 1.0
                for i in ti.static(range(dim)):
                    weight *= w[offset[i]][i]
                S_grid_v[base +
                         offset] += weight * (p_mass * F_v[p] + affine @ dpos)
                S_grid_m[base + offset] += weight * p_mass
                # 注释该部分代码实现单向的耦合，烟雾不会影响其他物质的运动
                # F_grid_v[base +
                #          offset] += weight * (p_mass * F_v[p] + affine @ dpos)
                # F_grid_m[base + offset] += weight * p_mass
        else:
            for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
                dpos = (offset - fx) * dx
                weight = 1.0
                for i in ti.static(range(dim)):
                    weight *= w[offset[i]][i]
                F_grid_v[base +
                         offset] += weight * (p_mass * F_v[p] + affine @ dpos)
                F_grid_m[base + offset] += weight * p_mass
                # 注释掉下面两行会失去烟雾的耦合，速度会提升10fps
                S_grid_v[base +
                         offset] += weight * (p_mass * F_v[p] + affine @ dpos)
                S_grid_m[base + offset] += weight * p_mass

    # Step2: Grid operations
    for I in ti.grouped(F_grid_m):
        if F_grid_m[I] > 0:
            F_grid_v[I] /= F_grid_m[I]
        F_grid_v[I] += dt * ti.Vector([g_x, g_y, g_z])
        cond = (I < bound) & (F_grid_v[I] < 0) | \
               (I > n_grid - bound) & (F_grid_v[I] > 0)
        F_grid_v[I] = ti.select(cond, 0, F_grid_v[I])

    for I in ti.grouped(S_grid_m):
        if S_grid_m[I] > 0:
            S_grid_v[I] /= S_grid_m[I]
        S_grid_v[I] -= dt * ti.Vector([g_x, g_y, g_z])
        if S_grid_v[I][1] < 0:
            S_grid_v[I][1] *= 0.8
        cond = (I < bound) & (S_grid_v[I] < 0) | \
               (I > n_grid - bound) & (S_grid_v[I] > 0)
        S_grid_v[I] = ti.select(cond, 0, S_grid_v[I])
    # Step3: Grid to particle (G2P)
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.zero(F_v[p])
        new_C = ti.zero(F_C[p])
        if F_materials[p] == SMOKE:
            for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
                dpos = (offset - fx) * dx
                weight = 1.0
                for i in ti.static(range(dim)):
                    weight *= w[offset[i]][i]
                g_v = S_grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        else:
            for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
                dpos = (offset - fx) * dx
                weight = 1.0
                for i in ti.static(range(dim)):
                    weight *= w[offset[i]][i]
                g_v = F_grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        F_v[p] = new_v
        F_x[p] += dt * F_v[p]
        F_C[p] = new_C

@ti.kernel
def set_color_by_material(mat_color: ti.types.ndarray()):
    for i in range(n_particles):
        mat = F_materials[i]
        F_colors[i] = ti.Vector(
            [mat_color[mat, 0], mat_color[mat, 1], mat_color[mat, 2], 1.0])



particles_radius = 0.003

material_colors = [(0.1, 0.6, 0.9), (0.93, 0.33, 0.23), (1.0, 1.0, 1.0), (0.5, 0.5, 0.5)]

@ti.kernel
def init():
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
        F_materials[i] = (i // group_size) # 0: fluid 1: jelly 2: snow 3:smoke




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
    init()
    show_options()
    while window.running:
        for _ in range(steps):
            substep(*GRAVITY)
        render()
        window.show()


if __name__ == '__main__':
    main()