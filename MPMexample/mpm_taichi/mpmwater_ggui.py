# MPM-MLS in 88 lines of Taichi code, originally created by @yuanming-hu
import taichi as ti
import src.material as material
import src.grid as grid

ti.init(arch=ti.gpu)

n_particles = 81920
n_grid = 128
dx = 1 / n_grid
dt = 2e-4

p_rho = 1  # particle density
p_vol = (dx * 0.5) ** 2  # particle volume
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3  # boundary condition
K = 400  # bulk modulus

x = ti.Vector.field(2, float, n_particles)
v = ti.Vector.field(2, float, n_particles)
C = ti.Matrix.field(2, 2, float, n_particles)  # affine matrix
J = ti.field(float, n_particles)  # 形变梯度F的行列式

grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))
grid_c = ti.field(float, (n_grid, n_grid))


@ti.kernel
def substep():
    for i, j in grid_m:  # clear former grid information
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:  # P2G (particle to grid)
        Xp = x[p] * n_grid
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # Lec8 P17
        fluid_pressure = K * (1 - J[p])
        cauchy_stress = fluid_pressure * ti.Matrix.identity(float, 2)
        # Lec8 P11
        force = -4 / dx ** 2 * p_vol * cauchy_stress  # 注意液体的情况下PK1数值上=cauchy_stress，因为默认液滴的形变梯度为J*I(identity matrix),还差一个wi（xi-xp）放到循环中计算
        affine = p_mass * C[p] - dt * force
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx  # 注意affine是在连续场中定义和完成计算的，所以dpos需要乘上dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    for I in ti.grouped(grid_m):
        grid_c[I] = material.Curl_cal2D(I, grid_v)
    for I in ti.grouped(grid_m):
        grid_v[I] += dt * material.Vorticity_cal2D(I, grid_c)

    for i, j in grid_m:
        if grid_m[i, j] > 0:  # normalization
            grid_v[i, j] /= grid_m[i, j]
        grid_v[i, j].y -= dt * gravity
        if i < bound and grid_v[i, j].x < 0:  # boundary conditon
            grid_v[i, j].x = 0
        if i > n_grid - bound and grid_v[i, j].x > 0:
            grid_v[i, j].x = 0
        if j < bound and grid_v[i, j].y < 0:
            grid_v[i, j].y = 0
        if j > n_grid - bound and grid_v[i, j].y > 0:
            grid_v[i, j].y = 0
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += (4 / dx ** 2) * weight * g_v.outer_product(dpos)  # (4 / dx ** 2) 指的是插值方式，见APIC论文 5.3 最后一段
        v[p] = new_v
        x[p] += dt * v[p]
        J[p] *= 1 + dt * new_C.trace()  # C矩阵c00,c11会导致体积发生变化
        C[p] = new_C


@ti.kernel
def init():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.4 + 0.3, ti.random() * 0.4 + 0.3]
        v[i] = [0, 0]
        J[i] = 1


def main():
    init()
    window = ti.ui.Window('MPM88', res=(512, 512), vsync=True)
    canvas = window.get_canvas()
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
        for s in range(50):
            substep()
        canvas.set_background_color(color=(1, 1, 1))
        canvas.circles(x, radius=0.004, color=(0.39, 0.772, 1.))
        window.show()
    # gui = ti.GUI('MPM88',(500, 500))
    # while gui.running and not gui.get_event(gui.ESCAPE):
    #     for s in range(50):
    #         substep()
    #     gui.clear(0x112F41)
    #     gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
    #     gui.show()


if __name__ == '__main__':
    main()
