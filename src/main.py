import numpy as np
import taichi as ti
import taichi.math as tm
ti.init(arch=ti.gpu)

##################################
# set simulation parameters
##################################
quality = 1
n_particles,n_grid = 9000 * quality**2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
gravity = [0, -9.8, 0]
bound = 3 # boundary condition
E, nu = 1.e3, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

WATER = 0
JELLY = 1
SNOW = 2
material_colors = [(0.1, 0.6, 0.9), (0.93, 0.33, 0.23), (1.0, 1.0, 1.0)]
##################################
# set simulation data structure
##################################
dim = 3
# 粒子态数据结构用大写开头
X = ti.Vector.field(dim, float, n_particles)
V = ti.Vector.field(dim, float, n_particles)
C = ti.Matrix.field(dim, dim, float, n_particles)  # affine speed matrix
F = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles)  # deformation gradient
Jp = ti.field(float, n_particles) # 塑性形变JP，初始是一个Identification Matrix
Material = ti.field(int, n_particles)
Particle_colors = ti.Vector.field(4, float, n_particles)
particles_radius = 0.005

grid_v = ti.Vector.field(dim, float, (n_grid, ) * dim)
grid_m = ti.field(float, (n_grid, ) * dim)
neighbour = (3, ) * dim

##################################
# GGUI Render Setting
##################################
res = (1080, 720)
window = ti.ui.Window("Real MPM 3D", res, vsync=True)

canvas = window.get_canvas()
gui = window.get_gui()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.3, 0.5)
camera.fov(55)



@ti.kernel
def init_sim(): # init simulation parameters
    group_size = n_particles // 3
    for i in range(n_particles):
        X[i] = [
            ti.random() * 0.2 + 0.3 + 0.1 * (i // group_size),
            ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size),
            ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size)
        ]
        V[i] = ti.Vector([0, 0, 0])
        C[i] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        F[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        Jp[i] = 1
        Material[i] = i // group_size  # 0: fluid 1: jelly 2: snow

@ti.kernel
def init_material(mat_color: ti.types.ndarray()):
    for i in range(n_particles):
        mat = Material[i]
        Particle_colors[i] = ti.Vector(
            [mat_color[mat, 0], mat_color[mat, 1], mat_color[mat, 2], 1.0])
@ti.kernel
def substep():
    for i in ti.grouped(grid_m): # clear grid info
        grid_m[i] = 0
        grid_v[i] = ti.zero(grid_v[i])
    for p in X: # particle in cell
        Xp = X[p] * inv_dx
        base = (Xp - 0.5).cast(int)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

        F[p] = (ti.Matrix.identity(float, 3) + dt * C[p]) @ F[p]  # F[p]: deformation gradient update (GAMES201 Lec8 P8)
        h = ti.exp(10 * (1.0 - Jp[p]))
        if Material[p] == JELLY:  # jelly, make it softer
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if Material[p] == WATER:  # liquid
            mu = 0.0

        U_matrix , sig, V_matrix = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(3)):
            new_sig = sig[d, d]
            if Material[p] == SNOW:  # Snow
                new_sig = ti.min(ti.max(sig[d, d], 1 - 2.5e-2),
                                 1 + 4.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if Material[p] == WATER:
            # Reset deformation gradient to avoid numerical instability
            new_F = ti.Matrix.identity(float, 3)
            new_F[0, 0] = J
            F[p] = new_F
        elif Material[p] == SNOW:
            # Reconstruct elastic deformation gradient after plasticity
            F[p] = U_matrix @ sig @ V_matrix.transpose()
        stress = 2 * mu * (F[p] - U_matrix @ V_matrix.transpose()) @ F[p].transpose(
        ) + ti.Matrix.identity(float, 3) * la * J * (J - 1)
        stress = (-dt * p_vol * 4) * stress / dx**2
        affine = stress + p_mass * C[p]

        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            grid_v[base + offset] += weight * (p_mass * V[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:
            grid_v[I] /= grid_m[I]
        grid_v[I] += dt * ti.Vector(gravity)
        cond = (I < bound) & (grid_v[I] < 0) | \
               (I > n_grid - bound) & (grid_v[I] > 0)
        grid_v[I] = ti.select(cond, 0, grid_v[I])

    for p in X:
        Xp = X[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.zero(V[p])
        new_C = ti.zero(C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx ** 2
        V[p] = new_v
        X[p] += dt * V[p]
        C[p] = new_C

def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))


    scene.particles(X, per_vertex_color=Particle_colors, radius=particles_radius)

    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

    canvas.scene(scene)

def main():
    init_sim()
    init_material(np.array(material_colors, dtype=np.float32))
    while window.running:
        for s in range(int(2e-3 // dt)):
            substep()
        render()
        window.show()
if __name__ == '__main__':
    main()