import taichi as ti
import taichi.math as tm

@ti.func
def boundary_sticky(pos, grid_v, n_grid, bound):
    cond = (pos < bound) & (grid_v[pos] < 0) | \
           (pos > n_grid - bound) & (grid_v[pos] > 0)
    grid_v[pos] = ti.select(cond, 0, grid_v[pos])


@ti.func
def boundary_slip(pos, grid_v, n_grid, bound):
    if pos[0] < bound:
        grid_v[pos][0] = 0
    if pos[0] > n_grid - bound:
        grid_v[pos][0] = 0
    if pos[1] < bound:
        grid_v[pos][1] = 0
    if pos[1] > n_grid - bound:
        grid_v[pos][1] = 0
    if pos[2] < bound:
        grid_v[pos][2] = 0
    if pos[2] > n_grid - bound:
        grid_v[pos][2] = 0

@ti.func
def boundary_separate(pos, grid_v, n_grid, bound):
    if pos[0] < bound and grid_v[pos][0] < 0:
        grid_v[pos][0] = 0
    if pos[0] > n_grid - bound and grid_v[pos][0] > 0:
        grid_v[pos][0] = 0
    if pos[1] < bound and grid_v[pos][1] < 0:
        grid_v[pos][1] = 0
    if pos[1] > n_grid - bound and grid_v[pos][1] > 0:
        grid_v[pos][1] = 0
    if pos[2] < bound and grid_v[pos][2] < 0:
        grid_v[pos][2] = 0
    if pos[2] > n_grid - bound and grid_v[pos][2] > 0:
        grid_v[pos][2] = 0
