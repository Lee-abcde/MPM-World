import taichi as ti
import taichi.math as tm
ti.init(arch=ti.gpu)  # Try to run on GPU

quality = 1  # Use a larger value for higher-res simulations
n_particles, n_grid = 9000 * quality**2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3 # boundary condition
E, nu = 0.1e4, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters


x = ti.Vector.field(2, dtype=float, shape=n_particles)
v = ti.Vector.field(2, dtype=float, shape=n_particles)
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  #为了模拟弹性和弹塑性，塑性材料需要记录这个物理信息
material = ti.field(dtype=int, shape=n_particles)  # material id
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation: 记录塑性形变的部分F矩阵

grid_v = ti.Vector.field(2, dtype=float,shape=(n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass


@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0

    for p in x:  # Particle state update and scatter to grid (P2G)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p] # F[p]: deformation gradient update (GAMES201 Lec8 P8)
        # handle lame parameters from different materials
        h = ti.exp(10 * (1.0 - Jp[p]))  # h: Hardening coefficient: snow gets harder when compressed
        if material[p] == 1:  # jelly, make it softer
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if material[p] == 0:  # liquid
            mu = 0.0

        U, sig, V = ti.svd(F[p])
        J = tm.determinant(F[p])
        if material[p] == 0:
            # F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
            # F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(F[p][0, 0] * F[p][1, 1])
            F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J) # Reset deformation gradient to avoid numerical instability
        elif material[p] == 2:
            J = 1
            for d in ti.static(range(2)):
                new_sig = sig[d, d]
                if material[p] == 2:  # Snow
                    new_sig = tm.clamp(sig[d, d], 1 - 2.5e-2, 1 + 4.5e-3)  # Plasticity Clamp Lec8 P22 Step3
                Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            # Reconstruct elastic deformation gradient after plasticity
            F[p] = U @ sig @ V.transpose()
        # corotated model R = U * V^T
        PK1_stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1) # calcuate Lec8 P11 (9) P*F^T 部分计算
        stress = (- 4 * inv_dx * inv_dx * p_vol ) * PK1_stress # Computing node force (GAMES201 Lec8 P11) 与（xi-xp)相乘是在循环中完成的
        affine = dt * stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]   # Momentum to velocity
            grid_v[i, j][1] -= dt * gravity  # gravity
            if i < bound and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0  # Boundary conditions
            if i > n_grid - bound and grid_v[i, j][0] > 0:
                grid_v[i, j][0] = 0
            if j < bound and grid_v[i, j][1] < 0:
                grid_v[i, j][1] = 0
            if j > n_grid - bound and grid_v[i, j][1] > 0:
                grid_v[i, j][1] = 0

    for p in x:  # grid to particle (G2P)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):  # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection


@ti.kernel
def initialize():
    group_size = n_particles // 3
    for i in range(n_particles):
        x[i] = [
            ti.random() * 0.2 + 0.3 + 0.1 * (i // group_size),
            ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size)
        ]
        material[i] = i // group_size  # 0: fluid 1: jelly 2: snow
        v[i] = ti.Matrix([0, 0])
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1


def main():
    initialize()
    window = ti.ui.Window('Taichi MLS-MPM-99', res = (512, 512), vsync=True)
    canvas = window.get_canvas()
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
        for s in range(int(2e-3 // dt)):
            substep()
        canvas.set_background_color(color=(1, 1, 1))
        canvas.circles(x, radius=0.004, color=(0.39, 0.772, 1.))
        window.show()
    # gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)
    # while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    #     for s in range(int(2e-3 // dt)):
    #         substep()
    #     gui.circles(x.to_numpy(),
    #                 radius=1.5,
    #                 palette=[0x068587, 0xED553B, 0xEEEEF0],
    #                 palette_indices=material)
    #     # Change to gui.show(f'{frame:06d}.png') to write images to disk
    #     gui.show()


if __name__ == '__main__':
    main()