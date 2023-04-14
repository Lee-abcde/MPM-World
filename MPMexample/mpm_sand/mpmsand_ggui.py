import taichi as ti

ti.init(arch=ti.vulkan)

quality = 1
n_particles = 20000 * quality ** 2
n_s_particles = ti.field(dtype = int, shape = ())
n_w_particles = ti.field(dtype = int, shape = ())
n_grid = 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 2e-4 / quality
gravity = ti.Vector([0, -9.8])
d = 2 # 没懂d是干什么的

# sand particle properties
x_s = ti.Vector.field(2, dtype = float, shape = n_particles) # position
v_s = ti.Vector.field(2, dtype = float, shape = n_particles) # velocity
C_s = ti.Matrix.field(2, 2, dtype = float, shape = n_particles) # affine velocity matrix
F_s = ti.Matrix.field(2, 2, dtype = float, shape = n_particles) # deformation gradient
phi_s = ti.field(dtype = float, shape = n_particles) # cohesion and saturation
c_C0 = ti.field(dtype = float, shape = n_particles) # initial cohesion (as maximum)
vc_s = ti.field(dtype = float, shape = n_particles) # tracks changes in the log of the volume gained during extension
alpha_s = ti.field(dtype = float, shape = n_particles) # yield surface size
q_s = ti.field(dtype = float, shape = n_particles) # harding state

# sand grid properties
grid_sv = ti.Vector.field(2, dtype = float, shape = (n_grid, n_grid)) # grid node momentum/velocity
grid_sm = ti.field(dtype = float, shape = (n_grid, n_grid)) # grid node mass
grid_sf = ti.Vector.field(2, dtype = float, shape = (n_grid, n_grid)) # forces in the sand


# constant values
p_vol, s_rho = (dx * 0.5) ** 2, 400
s_mass= p_vol * s_rho

E_s, nu_s = 3.537e5, 0.3 # sand's Young's modulus and Poisson's ratio
mu_s, lambda_s = E_s / (2 * (1 + nu_s)), E_s * nu_s / ((1 + nu_s) * (1 - 2 * nu_s)) # sand's Lame parameters

mu_b = 0.75 # coefficient of friction

state = ti.field(dtype = int, shape = n_particles)
pi = 3.14159265358979
@ti.func
def project(e0, p):
    e = e0 + vc_s[p] / d * ti.Matrix.identity(float, 2) # volume correction treatment
    e += (c_C0[p] * (1.0 - phi_s[p])) / (d * alpha_s[p]) * ti.Matrix.identity(float, 2) # effects of cohesion
    ehat = e - e.trace() / d * ti.Matrix.identity(float, 2)
    Fnorm = ti.sqrt(ehat[0, 0] ** 2 + ehat[1, 1] ** 2) # Frobenius norm
    yp = Fnorm + (d * lambda_s + 2 * mu_s) / (2 * mu_s) * e.trace() * alpha_s[p] # delta gamma
    new_e = ti.Matrix.zero(float, 2, 2)
    delta_q = 0.0
    if Fnorm <= 0 or e.trace() > 0: # Case II:
        new_e = ti.Matrix.zero(float, 2, 2)
        delta_q = ti.sqrt(e[0, 0] ** 2 + e[1, 1] ** 2)
        state[p] = 0
    elif yp <= 0: # Case I:
        new_e = e0 # return initial matrix without volume correction and cohesive effect
        delta_q = 0
        state[p] = 1
    else: # Case III:
        new_e = e - yp / Fnorm * ehat
        delta_q = yp
        state[p] = 2

    return new_e, delta_q

h0, h1, h2, h3 = 35, 9, 0.2, 10
@ti.func
def hardening(dq, p): # The amount of hardening depends on the amount of correction that occurred due to plasticity
    q_s[p] += dq
    phi = h0 + (h1 * q_s[p] - h3) * ti.exp(-h2 * q_s[p])
    phi = phi / 180 * pi # details in Table. 3: Friction angle phi_F and hardening parameters h0, h1, and h3 are listed in degrees for convenience
    sin_phi = ti.sin(phi)
    alpha_s[p] = ti.sqrt(2 / 3) * (2 * sin_phi) / (3 - sin_phi)

@ti.kernel
def substep():
    # set zero initial state for both water/sand grid
    for i, j in grid_sm:
        grid_sv[i, j] = [0, 0]
        grid_sm[i, j] = 0
        grid_sf[i, j] = [0, 0]

    # P2G (sand's part)
    # for p in range(n_s_particles):
    for p in x_s:
        base = (x_s[p] * inv_dx - 0.5).cast(int)
        # 下面的判断防止的边角的沙子产生爆炸效果
        # if base[0] < 0 or base[1] < 0 or base[0] >= n_grid - 2 or base[1] >= n_grid - 2:
        #     continue
        fx = x_s[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        U, sig, V = ti.svd(F_s[p])
        inv_sig = sig.inverse()
        e = ti.Matrix([[ti.log(sig[0, 0]), 0], [0, ti.log(sig[1, 1])]])
        stress = U @ (2 * mu_s * inv_sig @ e + lambda_s * e.trace() * inv_sig) @ V.transpose() # formula (25)
        stress = (-p_vol * 4 * inv_dx * inv_dx) * stress @ F_s[p].transpose()
        # stress *= h(e)
        # print(h(e))
        affine = s_mass * C_s[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_sv[base + offset] += weight * (s_mass * v_s[p] + affine @ dpos)
            grid_sm[base + offset] += weight * s_mass
            grid_sf[base + offset] += weight * stress @ dpos

    # Update Grids Momentum
    for i, j in grid_sm:
        if grid_sm[i, j] > 0:
            grid_sv[i, j] = (1 / grid_sm[i, j]) * grid_sv[i, j] # Momentum to velocity

        grid_sv[i, j] += dt * (gravity + grid_sf[i, j] / grid_sm[i, j]) # Update explicit force

        normal = ti.Vector.zero(float, 2)
        if grid_sm[i, j] > 0:
            if i < 3 and grid_sv[i, j][0] < 0:          normal = ti.Vector([1, 0])
            if i > n_grid - 3 and grid_sv[i, j][0] > 0: normal = ti.Vector([-1, 0])
            if j < 3 and grid_sv[i, j][1] < 0:          normal = ti.Vector([0, 1])
            if j > n_grid - 3 and grid_sv[i, j][1] > 0: normal = ti.Vector([0, -1])
        if not (normal[0] == 0 and normal[1] == 0): # Apply friction
            s = grid_sv[i, j].dot(normal)
            if s <= 0:
                v_normal = s * normal
                v_tangent = grid_sv[i, j] - v_normal # divide velocity into normal and tangential parts
                vt = v_tangent.norm()
                if vt > 1e-12: grid_sv[i, j] = v_tangent - (vt if vt < -mu_b * s else -mu_b * s) * (v_tangent / vt) # The Coulomb friction law

    # G2P (sand's part)
    # for p in range(n_s_particles):
    for p in x_s:
        base = (x_s[p] * inv_dx - 0.5).cast(int)
        if base[0] < 0 or base[1] < 0 or base[0] >= n_grid - 2 or base[1] >= n_grid - 2:
            continue
        fx = x_s[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)): # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_sv[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)

        F_s[p] = (ti.Matrix.identity(float, 2) + dt * new_C) @ F_s[p]
        v_s[p], C_s[p] = new_v, new_C
        x_s[p] += dt * v_s[p]

        U, sig, V = ti.svd(F_s[p])
        e = ti.Matrix([[ti.log(sig[0, 0]), 0], [0, ti.log(sig[1, 1])]])
        new_e, dq = project(e, p)
        hardening(dq, p)
        new_F = U @ ti.Matrix([[ti.exp(new_e[0, 0]), 0], [0, ti.exp(new_e[1, 1])]]) @ V.transpose()
        vc_s[p] += -ti.log(new_F.determinant()) + ti.log(F_s[p].determinant()) # formula (26)
        F_s[p] = new_F

@ti.kernel
def initialize():
    n_s_particles[None] = 10000 * quality ** 2
    for i in x_s:
        x_s[i] = [ti.random() * 0.25 + 0.4, ti.random() * 0.4 + 0.2]
        v_s[i] = ti.Matrix([0, 0])
        F_s[i] = ti.Matrix([[1, 0], [0, 1]])
        c_C0[i] = -0.01
        alpha_s[i] = 0.267765

    n_w_particles[None] = 0

initialize()
window = ti.ui.Window('Window Title', res = (512, 512), pos = (150, 150))
canvas = window.get_canvas()
while window.running:
    for s in range(50):
        substep()

    canvas.set_background_color((1., 1., 1.))
    canvas.circles(x_s, radius=0.002, color=(0.4, 0.3, 0.2))
    window.show()
