"""
Post-processing module for 2D FEM elastic analysis.

Provides 19 functions following the naming convention:
    {target}_{mode}

target:  displacement | stress | principal_stress | mises
mode:    point | line | max | min | absmin

Calling convention:
    target_mode(location, grid_path, disp_path, E=..., niu=...)

    - point:   target_point(x, y, grid_path, disp_path, ...)
    - line:    target_line(x1, y1, x2, y2, grid_path, disp_path, ...)
               -> saves a plot to case/ and returns data arrays
    - max:     target_max(grid_path, disp_path, ...)
    - min:     target_min(grid_path, disp_path, ...)

After running E_main.py, simply import from test.py and call directly.
"""

import numpy as np
import matplotlib.pyplot as plt
from A_module import Vertex, Graph
from Y_get_info import load_grid_from_npz
from G_get_displacement import interpolate_displacement, load_result
from F_find_tri import find_triangle_of_point
from B_calcu_metrix import getD, LN, N_ijk


# ============================================================
# Internal helpers
# ============================================================

def _load(grid_path, disp_path):
    """Load graph and displacement vector from result files."""
    g, U = load_result(grid_path, disp_path)
    return g, U


def _stress_at_point(g, U, x, y, E, niu):
    """Compute (sigma_x, sigma_y, tau_xy) at a point. Returns None if outside mesh."""
    tri = find_triangle_of_point(g, x, y)
    if tri is None:
        return None
    i1, i2, i3, *_ = tri
    v1, v2, v3 = g.vertices[i1], g.vertices[i2], g.vertices[i3]
    DB = np.dot(getD(E, niu, is_stress=1), LN(N_ijk(v1, v2, v3)))
    tri_d = [U[2*i1], U[2*i1+1], U[2*i2], U[2*i2+1], U[2*i3], U[2*i3+1]]
    sx, sy, txy = np.dot(DB, tri_d)
    return sx, sy, txy


def _principal(sx, sy, txy):
    """Compute principal stresses from (sigma_x, sigma_y, tau_xy)."""
    avg = (sx + sy) * 0.5
    r = np.sqrt(((sx - sy) * 0.5) ** 2 + txy ** 2)
    s1 = avg + r
    s2 = avg - r
    return s1, s2


def _mises(sx, sy, txy):
    """Compute von Mises stress."""
    return np.sqrt(sx**2 + sy**2 - sx*sy + 3.0 * txy**2)


def _sample_line(g, U, x1, y1, x2, y2, n_samples, E, niu, value_fn):
    """
    Sample a scalar field along a line.

    value_fn: (g, U, x, y, E, niu) -> float or None
    Returns: array of length n_samples (nan where outside mesh)
    """
    xs = np.linspace(x1, x2, n_samples)
    ys = np.linspace(y1, y2, n_samples)
    vals = np.full(n_samples, np.nan)
    for k in range(n_samples):
        v = value_fn(g, U, xs[k], ys[k], E, niu)
        if v is not None:
            vals[k] = v
    return xs, ys, vals


def _sample_line_multi(g, U, x1, y1, x2, y2, n_samples, E, niu, value_fn):
    """
    Sample a field along a line. Handles both scalar and tuple returns.

    value_fn: (g, U, x, y, E, niu) -> float | tuple | None
    Returns: xs, ys, [array1, array2, ...]
    """
    xs = np.linspace(x1, x2, n_samples)
    ys = np.linspace(y1, y2, n_samples)
    n_comps = None
    result_lists = None
    for k in range(n_samples):
        v = value_fn(g, U, xs[k], ys[k], E, niu)
        if v is not None:
            if not isinstance(v, (tuple, list, np.ndarray)):
                v = (v,)  # wrap scalar
            if n_comps is None:
                n_comps = len(v)
                result_lists = [[] for _ in range(n_comps)]
            for i in range(n_comps):
                result_lists[i].append(v[i])
        else:
            if result_lists is not None:
                for lst in result_lists:
                    lst.append(np.nan)
    if result_lists is None:
        empty = np.full(n_samples, np.nan)
        return xs, ys, [empty]
    return xs, ys, [np.array(lst) for lst in result_lists]


def _line_plot(xs, ys, data_arrays, labels, title, filename):
    """Save a line-plot figure with one or more curves."""
    t = np.linspace(0, 1, len(xs))
    fig, ax = plt.subplots(figsize=(10, 5))
    for arr, lbl in zip(data_arrays, labels):
        ax.plot(t, arr, linewidth=1.5, label=lbl)
    ax.set_xlabel('Normalized distance along line')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(data_arrays) > 1:
        ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {filename}")


def _line_plot_multi(xs, ys, data_arrays, labels, title, filename):
    """Save a multi-subplot line-plot figure."""
    n = len(data_arrays)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    t = np.linspace(0, 1, len(xs))
    for i, (arr, lbl) in enumerate(zip(data_arrays, labels)):
        axes[i].plot(t, arr, linewidth=1.5)
        axes[i].set_ylabel(lbl)
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel('Normalized distance along line')
    axes[0].set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {filename}")


def _global_extremum(g, U, E, niu, value_fn, mode, label):
    """
    Find global max / min / absmin of a scalar quantity over all elements.

    mode: 'max' | 'min' | 'absmin'
      - max:    largest algebraic value
      - min:    smallest algebraic value
      - absmin: smallest absolute value (closest to zero)

    value_fn: (g, U, x, y, E, niu) -> float
    """
    if mode == 'absmin':
        best_val = float('inf')   # tracks smallest abs(value)
    elif mode == 'max':
        best_val = -float('inf')
    else:
        best_val = float('inf')

    best_info = None

    for tri in g.triangles:
        i1, i2, i3 = tri[0], tri[1], tri[2]
        v1, v2, v3 = g.vertices[i1], g.vertices[i2], g.vertices[i3]
        cx = (v1.x + v2.x + v3.x) / 3.0
        cy = (v1.y + v2.y + v3.y) / 3.0
        val = value_fn(g, U, cx, cy, E, niu)
        if val is None:
            continue
        if mode == 'absmin':
            if abs(val) < best_val:
                best_val = abs(val)
                best_info = (cx, cy, val, v1, v2, v3)
        elif (mode == 'max' and val > best_val) or (mode == 'min' and val < best_val):
            best_val = val
            best_info = (cx, cy, val, v1, v2, v3)

    if best_info is None:
        print(f"No valid {label} found.")
        return None

    cx, cy, val, v1, v2, v3 = best_info
    print(f"Global {mode} {label}: {val:.6e} at element centroid ({cx:.4f}, {cy:.4f})")
    print(f"  vertices: ({v1.x:.4f}, {v1.y:.4f})  ({v2.x:.4f}, {v2.y:.4f})  ({v3.x:.4f}, {v3.y:.4f})")
    return cx, cy, val


# ============================================================
# Public API — displacement
# ============================================================

def displacement_point(x, y, grid_path='case/grid.npz',
                       disp_path='case/DisplacementResult.npz'):
    """Print displacement (u, v) at a given point."""
    g, U = _load(grid_path, disp_path)
    uv = interpolate_displacement(g, U, x, y)
    if uv is None:
        print(f"Point ({x}, {y}) is outside the mesh.")
        return None, None
    u, v = uv
    print(f"Displacement at ({x}, {y}): u = {u:.6e}, v = {v:.6e}")
    return u, v


def displacement_line(x1, y1, x2, y2,
                      grid_path='case/grid.npz',
                      disp_path='case/DisplacementResult.npz',
                      n_samples=100):
    """
    Plot displacement components (u, v) along a line from (x1,y1) to (x2,y2).
    Saves the figure to case/ and returns (u_arr, v_arr).
    """
    g, U = _load(grid_path, disp_path)

    def _disp_uv(g, U, x, y, E, niu):
        return interpolate_displacement(g, U, x, y)

    xs, ys, arrs = _sample_line_multi(g, U, x1, y1, x2, y2, n_samples,
                                      200e9, 0.3, _disp_uv)
    if len(arrs) < 2:
        return None, None
    u_arr, v_arr = arrs[0], arrs[1]

    fname = f'case/displacement_line_{x1}_{y1}_{x2}_{y2}.png'
    _line_plot_multi(xs, ys, [u_arr, v_arr],
                     ['u (x-displacement)', 'v (y-displacement)'],
                     f'Displacement along ({x1},{y1}) → ({x2},{y2})', fname)
    return u_arr, v_arr


def displacement_max(grid_path='case/grid.npz',
                     disp_path='case/DisplacementResult.npz'):
    """Print global maximum displacement magnitude over all nodes."""
    g, U = _load(grid_path, disp_path)
    best_mag = 0.0
    best_info = None
    for v in g.vertices:
        ux, uy = U[2*v.idx], U[2*v.idx+1]
        mag = np.sqrt(ux**2 + uy**2)
        if mag > best_mag:
            best_mag = mag
            best_info = (v.idx, v.x, v.y, ux, uy, mag)
    if best_info is None:
        print("No displacement data found.")
        return None
    idx, x, y, ux, uy, mag = best_info
    print(f"Global max displacement: |u| = {mag:.6e} at node {idx}"
          f" ({x:.4f}, {y:.4f}), u = {ux:.6e}, v = {uy:.6e}")
    return best_info


def displacement_min(grid_path='case/grid.npz',
                     disp_path='case/DisplacementResult.npz'):
    """Print global minimum non-zero displacement magnitude over all nodes."""
    g, U = _load(grid_path, disp_path)
    best_mag = float('inf')
    best_info = None
    for v in g.vertices:
        ux, uy = U[2*v.idx], U[2*v.idx+1]
        mag = np.sqrt(ux**2 + uy**2)
        if mag < best_mag:
            best_mag = mag
            best_info = (v.idx, v.x, v.y, ux, uy, mag)
    if best_info is None:
        print("No displacement data found.")
        return None
    idx, x, y, ux, uy, mag = best_info
    print(f"Global min displacement: |u| = {mag:.6e} at node {idx}"
          f" ({x:.4f}, {y:.4f}), u = {ux:.6e}, v = {uy:.6e}")
    return best_info


# ============================================================
# Public API — stress (sigma_x, sigma_y, tau_xy)
# ============================================================

def stress_point(x, y,
                 grid_path='case/grid.npz',
                 disp_path='case/DisplacementResult.npz',
                 E=200e9, niu=0.3):
    """Print stress (sigma_x, sigma_y, tau_xy) at a given point."""
    g, U = _load(grid_path, disp_path)
    s = _stress_at_point(g, U, x, y, E, niu)
    if s is None:
        print(f"Point ({x}, {y}) is outside the mesh.")
        return None, None, None
    sx, sy, txy = s
    print(f"Stress at ({x}, {y}): sigma_x = {sx:.6e}, sigma_y = {sy:.6e},"
          f" tau_xy = {txy:.6e}")
    return sx, sy, txy


def stress_line(x1, y1, x2, y2,
                grid_path='case/grid.npz',
                disp_path='case/DisplacementResult.npz',
                E=200e9, niu=0.3, n_samples=100):
    """
    Plot stress components along a line from (x1,y1) to (x2,y2).
    Saves figure to case/ and returns (sx_arr, sy_arr, txy_arr).
    """
    g, U = _load(grid_path, disp_path)

    def _stress_3(g, U, x, y, E, niu):
        s = _stress_at_point(g, U, x, y, E, niu)
        return s  # (sx, sy, txy) or None

    xs, ys, arrs = _sample_line_multi(g, U, x1, y1, x2, y2, n_samples,
                                      E, niu, _stress_3)
    if len(arrs) < 3:
        return None, None, None
    sx_arr, sy_arr, txy_arr = arrs[0], arrs[1], arrs[2]

    fname = f'case/stress_line_{x1}_{y1}_{x2}_{y2}.png'
    _line_plot_multi(xs, ys, [sx_arr, sy_arr, txy_arr],
                     ['sigma_x', 'sigma_y', 'tau_xy'],
                     f'Stress along ({x1},{y1}) → ({x2},{y2})', fname)
    return sx_arr, sy_arr, txy_arr


def stress_max(grid_path='case/grid.npz',
               disp_path='case/DisplacementResult.npz',
               E=200e9, niu=0.3):
    """Print global maximum of each stress component over all elements."""
    g, U = _load(grid_path, disp_path)
    best_sx = -float('inf'); best_sy = -float('inf'); best_txy = -float('inf')
    info_sx = info_sy = info_txy = None
    for tri in g.triangles:
        i1, i2, i3 = tri[0], tri[1], tri[2]
        v1, v2, v3 = g.vertices[i1], g.vertices[i2], g.vertices[i3]
        cx = (v1.x + v2.x + v3.x) / 3.0
        cy = (v1.y + v2.y + v3.y) / 3.0
        s = _stress_at_point(g, U, cx, cy, E, niu)
        if s is None:
            continue
        sx, sy, txy = s
        if sx > best_sx:
            best_sx = sx; info_sx = (cx, cy, sx, v1, v2, v3)
        if sy > best_sy:
            best_sy = sy; info_sy = (cx, cy, sy, v1, v2, v3)
        if txy > best_txy:
            best_txy = txy; info_txy = (cx, cy, txy, v1, v2, v3)
    def _print_stress(label, val, info):
        cx, cy, _, v1, v2, v3 = info
        print(f"Global max {label}: {val:.6e} at element centroid ({cx:.4f}, {cy:.4f})")
        print(f"  vertices: ({v1.x:.4f}, {v1.y:.4f})  ({v2.x:.4f}, {v2.y:.4f})  ({v3.x:.4f}, {v3.y:.4f})")
    _print_stress('sigma_x', best_sx, info_sx)
    _print_stress('sigma_y', best_sy, info_sy)
    _print_stress('tau_xy',   best_txy, info_txy)
    return (best_sx, info_sx), (best_sy, info_sy), (best_txy, info_txy)


def stress_min(grid_path='case/grid.npz',
               disp_path='case/DisplacementResult.npz',
               E=200e9, niu=0.3):
    """Print global minimum of each stress component over all elements."""
    g, U = _load(grid_path, disp_path)
    best_sx = float('inf'); best_sy = float('inf'); best_txy = float('inf')
    info_sx = info_sy = info_txy = None
    for tri in g.triangles:
        i1, i2, i3 = tri[0], tri[1], tri[2]
        v1, v2, v3 = g.vertices[i1], g.vertices[i2], g.vertices[i3]
        cx = (v1.x + v2.x + v3.x) / 3.0
        cy = (v1.y + v2.y + v3.y) / 3.0
        s = _stress_at_point(g, U, cx, cy, E, niu)
        if s is None:
            continue
        sx, sy, txy = s
        if sx < best_sx:
            best_sx = sx; info_sx = (cx, cy, sx, v1, v2, v3)
        if sy < best_sy:
            best_sy = sy; info_sy = (cx, cy, sy, v1, v2, v3)
        if txy < best_txy:
            best_txy = txy; info_txy = (cx, cy, txy, v1, v2, v3)
    def _print_stress(label, val, info):
        cx, cy, _, v1, v2, v3 = info
        print(f"Global min {label}: {val:.6e} at element centroid ({cx:.4f}, {cy:.4f})")
        print(f"  vertices: ({v1.x:.4f}, {v1.y:.4f})  ({v2.x:.4f}, {v2.y:.4f})  ({v3.x:.4f}, {v3.y:.4f})")
    _print_stress('sigma_x', best_sx, info_sx)
    _print_stress('sigma_y', best_sy, info_sy)
    _print_stress('tau_xy',   best_txy, info_txy)
    return (best_sx, info_sx), (best_sy, info_sy), (best_txy, info_txy)


def stress_absmin(grid_path='case/grid.npz',
                   disp_path='case/DisplacementResult.npz',
                   E=200e9, niu=0.3):
    """Print global absolute-minimum (closest to zero) of each stress component."""
    g, U = _load(grid_path, disp_path)
    best_sx = float('inf'); best_sy = float('inf'); best_txy = float('inf')
    info_sx = info_sy = info_txy = None
    for tri in g.triangles:
        i1, i2, i3 = tri[0], tri[1], tri[2]
        v1, v2, v3 = g.vertices[i1], g.vertices[i2], g.vertices[i3]
        cx = (v1.x + v2.x + v3.x) / 3.0
        cy = (v1.y + v2.y + v3.y) / 3.0
        s = _stress_at_point(g, U, cx, cy, E, niu)
        if s is None:
            continue
        sx, sy, txy = s
        if abs(sx) < best_sx:
            best_sx = abs(sx); info_sx = (cx, cy, sx, v1, v2, v3)
        if abs(sy) < best_sy:
            best_sy = abs(sy); info_sy = (cx, cy, sy, v1, v2, v3)
        if abs(txy) < best_txy:
            best_txy = abs(txy); info_txy = (cx, cy, txy, v1, v2, v3)
    def _print_stress(label, val, info):
        cx, cy, _, v1, v2, v3 = info
        print(f"Global absmin {label}: {val:.6e} at element centroid ({cx:.4f}, {cy:.4f})")
        print(f"  vertices: ({v1.x:.4f}, {v1.y:.4f})  ({v2.x:.4f}, {v2.y:.4f})  ({v3.x:.4f}, {v3.y:.4f})")
    _print_stress('sigma_x', info_sx[2], info_sx)
    _print_stress('sigma_y', info_sy[2], info_sy)
    _print_stress('tau_xy',   info_txy[2], info_txy)
    return (best_sx, info_sx), (best_sy, info_sy), (best_txy, info_txy)


# ============================================================
# Public API — principal stress
# ============================================================

def _principal_at_point(g, U, x, y, E, niu):
    """Compute (sigma_1, sigma_2) at a point."""
    s = _stress_at_point(g, U, x, y, E, niu)
    if s is None:
        return None
    return _principal(*s)


def principal_stress_point(x, y,
                           grid_path='case/grid.npz',
                           disp_path='case/DisplacementResult.npz',
                           E=200e9, niu=0.3):
    """Print principal stresses (sigma_1, sigma_2) at a given point."""
    g, U = _load(grid_path, disp_path)
    p = _principal_at_point(g, U, x, y, E, niu)
    if p is None:
        print(f"Point ({x}, {y}) is outside the mesh.")
        return None, None
    s1, s2 = p
    print(f"Principal stress at ({x}, {y}): sigma_1 = {s1:.6e}, sigma_2 = {s2:.6e}")
    return s1, s2


def principal_stress_line(x1, y1, x2, y2,
                          grid_path='case/grid.npz',
                          disp_path='case/DisplacementResult.npz',
                          E=200e9, niu=0.3, n_samples=100):
    """
    Plot principal stresses along a line.
    Saves figure to case/ and returns (s1_arr, s2_arr).
    """
    g, U = _load(grid_path, disp_path)
    xs, ys, arrs = _sample_line_multi(g, U, x1, y1, x2, y2, n_samples,
                                      E, niu, _principal_at_point)
    if len(arrs) < 2:
        return None, None
    s1_arr, s2_arr = arrs[0], arrs[1]
    fname = f'case/principal_stress_line_{x1}_{y1}_{x2}_{y2}.png'
    _line_plot_multi(xs, ys, [s1_arr, s2_arr],
                     ['sigma_1', 'sigma_2'],
                     f'Principal stress along ({x1},{y1}) → ({x2},{y2})', fname)
    return s1_arr, s2_arr


def principal_stress_max(grid_path='case/grid.npz',
                         disp_path='case/DisplacementResult.npz',
                         E=200e9, niu=0.3):
    """Print global max of sigma_1 and sigma_2 over all elements."""
    g, U = _load(grid_path, disp_path)
    best_s1 = -float('inf'); best_s2 = -float('inf')
    info_s1 = info_s2 = None
    for tri in g.triangles:
        i1, i2, i3 = tri[0], tri[1], tri[2]
        v1, v2, v3 = g.vertices[i1], g.vertices[i2], g.vertices[i3]
        cx = (v1.x + v2.x + v3.x) / 3.0
        cy = (v1.y + v2.y + v3.y) / 3.0
        p = _principal_at_point(g, U, cx, cy, E, niu)
        if p is None:
            continue
        s1, s2 = p
        if s1 > best_s1:
            best_s1 = s1; info_s1 = (cx, cy, s1, v1, v2, v3)
        if s2 > best_s2:
            best_s2 = s2; info_s2 = (cx, cy, s2, v1, v2, v3)
    if info_s1 is None or info_s2 is None:
        print("No valid principal stress found.")
        return None, None
    def _print_ps(label, val, info):
        cx, cy, _, v1, v2, v3 = info
        print(f"Global max {label}: {val:.6e} at element centroid ({cx:.4f}, {cy:.4f})")
        print(f"  vertices: ({v1.x:.4f}, {v1.y:.4f})  ({v2.x:.4f}, {v2.y:.4f})  ({v3.x:.4f}, {v3.y:.4f})")
    _print_ps('sigma_1', best_s1, info_s1)
    _print_ps('sigma_2', best_s2, info_s2)
    return (best_s1, info_s1), (best_s2, info_s2)


def principal_stress_min(grid_path='case/grid.npz',
                         disp_path='case/DisplacementResult.npz',
                         E=200e9, niu=0.3):
    """Print global min of sigma_1 and sigma_2 over all elements."""
    g, U = _load(grid_path, disp_path)
    best_s1 = float('inf'); best_s2 = float('inf')
    info_s1 = info_s2 = None
    for tri in g.triangles:
        i1, i2, i3 = tri[0], tri[1], tri[2]
        v1, v2, v3 = g.vertices[i1], g.vertices[i2], g.vertices[i3]
        cx = (v1.x + v2.x + v3.x) / 3.0
        cy = (v1.y + v2.y + v3.y) / 3.0
        p = _principal_at_point(g, U, cx, cy, E, niu)
        if p is None:
            continue
        s1, s2 = p
        if s1 < best_s1:
            best_s1 = s1; info_s1 = (cx, cy, s1, v1, v2, v3)
        if s2 < best_s2:
            best_s2 = s2; info_s2 = (cx, cy, s2, v1, v2, v3)
    if info_s1 is None or info_s2 is None:
        print("No valid principal stress found.")
        return None, None
    def _print_ps(label, val, info):
        cx, cy, _, v1, v2, v3 = info
        print(f"Global min {label}: {val:.6e} at element centroid ({cx:.4f}, {cy:.4f})")
        print(f"  vertices: ({v1.x:.4f}, {v1.y:.4f})  ({v2.x:.4f}, {v2.y:.4f})  ({v3.x:.4f}, {v3.y:.4f})")
    _print_ps('sigma_1', best_s1, info_s1)
    _print_ps('sigma_2', best_s2, info_s2)
    return (best_s1, info_s1), (best_s2, info_s2)


def principal_stress_absmin(grid_path='case/grid.npz',
                             disp_path='case/DisplacementResult.npz',
                             E=200e9, niu=0.3):
    """Print global absolute-minimum of sigma_1 and sigma_2 over all elements."""
    g, U = _load(grid_path, disp_path)
    best_s1 = float('inf'); best_s2 = float('inf')
    info_s1 = info_s2 = None
    for tri in g.triangles:
        i1, i2, i3 = tri[0], tri[1], tri[2]
        v1, v2, v3 = g.vertices[i1], g.vertices[i2], g.vertices[i3]
        cx = (v1.x + v2.x + v3.x) / 3.0
        cy = (v1.y + v2.y + v3.y) / 3.0
        p = _principal_at_point(g, U, cx, cy, E, niu)
        if p is None:
            continue
        s1, s2 = p
        if abs(s1) < best_s1:
            best_s1 = abs(s1); info_s1 = (cx, cy, s1, v1, v2, v3)
        if abs(s2) < best_s2:
            best_s2 = abs(s2); info_s2 = (cx, cy, s2, v1, v2, v3)
    if info_s1 is None or info_s2 is None:
        print("No valid principal stress found.")
        return None, None
    def _print_ps(label, val, info):
        cx, cy, _, v1, v2, v3 = info
        print(f"Global absmin {label}: {val:.6e} at element centroid ({cx:.4f}, {cy:.4f})")
        print(f"  vertices: ({v1.x:.4f}, {v1.y:.4f})  ({v2.x:.4f}, {v2.y:.4f})  ({v3.x:.4f}, {v3.y:.4f})")
    _print_ps('sigma_1', info_s1[2], info_s1)
    _print_ps('sigma_2', info_s2[2], info_s2)
    return (best_s1, info_s1), (best_s2, info_s2)


# ============================================================
# Public API — von Mises stress
# ============================================================

def _mises_at_point(g, U, x, y, E, niu):
    """Compute von Mises stress at a point."""
    s = _stress_at_point(g, U, x, y, E, niu)
    if s is None:
        return None
    return _mises(*s)


def mises_point(x, y,
                grid_path='case/grid.npz',
                disp_path='case/DisplacementResult.npz',
                E=200e9, niu=0.3):
    """Print von Mises stress at a given point."""
    g, U = _load(grid_path, disp_path)
    vm = _mises_at_point(g, U, x, y, E, niu)
    if vm is None:
        print(f"Point ({x}, {y}) is outside the mesh.")
        return None
    print(f"Von Mises stress at ({x}, {y}): {vm:.6e}")
    return vm


def mises_line(x1, y1, x2, y2,
               grid_path='case/grid.npz',
               disp_path='case/DisplacementResult.npz',
               E=200e9, niu=0.3, n_samples=100):
    """
    Plot von Mises stress along a line.
    Saves figure to case/ and returns (mises_arr,).
    """
    g, U = _load(grid_path, disp_path)

    def _vm(g, U, x, y, E, niu):
        return _mises_at_point(g, U, x, y, E, niu)

    xs, ys, arrs = _sample_line_multi(g, U, x1, y1, x2, y2, n_samples,
                                      E, niu, _vm)
    mises_arr = arrs[0] if arrs else np.full(n_samples, np.nan)
    fname = f'case/mises_line_{x1}_{y1}_{x2}_{y2}.png'
    _line_plot_multi(xs, ys, [mises_arr], ['Von Mises'],
                     f'Von Mises stress along ({x1},{y1}) → ({x2},{y2})', fname)
    return mises_arr


def mises_max(grid_path='case/grid.npz',
              disp_path='case/DisplacementResult.npz',
              E=200e9, niu=0.3):
    """Print global maximum von Mises stress over all elements."""
    g, U = _load(grid_path, disp_path)
    return _global_extremum(g, U, E, niu, _mises_at_point, 'max', 'Von Mises')


def mises_min(grid_path='case/grid.npz',
              disp_path='case/DisplacementResult.npz',
              E=200e9, niu=0.3):
    """Print global minimum von Mises stress over all elements."""
    g, U = _load(grid_path, disp_path)
    return _global_extremum(g, U, E, niu, _mises_at_point, 'min', 'Von Mises')


def mises_absmin(grid_path='case/grid.npz',
                  disp_path='case/DisplacementResult.npz',
                  E=200e9, niu=0.3):
    """Print global absolute-minimum (non-zero) von Mises stress over all elements."""
    g, U = _load(grid_path, disp_path)
    return _global_extremum(g, U, E, niu, _mises_at_point, 'absmin', 'Von Mises')


# ============================================================
# Convenience: run all post-processing at once
# ============================================================

def run_all(x_point=None, y_point=None,
            line_start=None, line_end=None,
            grid_path='case/grid.npz',
            disp_path='case/DisplacementResult.npz',
            E=200e9, niu=0.3):
    """
    Run all post-processing steps on a saved FEM result.

    Parameters
    ----------
    x_point, y_point : float, optional
        Point for point-wise evaluation. Default uses (30, 1.5).
    line_start : tuple (x, y), optional
    line_end : tuple (x, y), optional
        Endpoints for line evaluation. Default uses ((0, 0), (30, 3)).
    """
    if x_point is None:
        x_point, y_point = 30.0, 1.5
    if line_start is None:
        line_start = (0.0, 0.0)
    if line_end is None:
        line_end = (30.0, 3.0)
    x1, y1 = line_start
    x2, y2 = line_end

    print("=" * 60)
    print("Displacement — point")
    print("=" * 60)
    displacement_point(x_point, y_point, grid_path, disp_path)

    print()
    print("=" * 60)
    print("Displacement — line")
    print("=" * 60)
    displacement_line(x1, y1, x2, y2, grid_path, disp_path)

    print()
    print("=" * 60)
    print("Displacement — max / min")
    print("=" * 60)
    displacement_max(grid_path, disp_path)
    displacement_min(grid_path, disp_path)

    print()
    print("=" * 60)
    print("Stress — point")
    print("=" * 60)
    stress_point(x_point, y_point, grid_path, disp_path, E, niu)

    print()
    print("=" * 60)
    print("Stress — line")
    print("=" * 60)
    stress_line(x1, y1, x2, y2, grid_path, disp_path, E, niu)

    print()
    print("=" * 60)
    print("Stress — max / min / absmin")
    print("=" * 60)
    stress_max(grid_path, disp_path, E, niu)
    stress_min(grid_path, disp_path, E, niu)
    stress_absmin(grid_path, disp_path, E, niu)

    print()
    print("=" * 60)
    print("Principal stress — point")
    print("=" * 60)
    principal_stress_point(x_point, y_point, grid_path, disp_path, E, niu)

    print()
    print("=" * 60)
    print("Principal stress — line")
    print("=" * 60)
    principal_stress_line(x1, y1, x2, y2, grid_path, disp_path, E, niu)

    print()
    print("=" * 60)
    print("Principal stress — max / min / absmin")
    print("=" * 60)
    principal_stress_max(grid_path, disp_path, E, niu)
    principal_stress_min(grid_path, disp_path, E, niu)
    principal_stress_absmin(grid_path, disp_path, E, niu)

    print()
    print("=" * 60)
    print("Von Mises — point")
    print("=" * 60)
    mises_point(x_point, y_point, grid_path, disp_path, E, niu)

    print()
    print("=" * 60)
    print("Von Mises — line")
    print("=" * 60)
    mises_line(x1, y1, x2, y2, grid_path, disp_path, E, niu)

    print()
    print("=" * 60)
    print("Von Mises — max / min / absmin")
    print("=" * 60)
    mises_max(grid_path, disp_path, E, niu)
    mises_min(grid_path, disp_path, E, niu)
    mises_absmin(grid_path, disp_path, E, niu)

    print()
    print("All post-processing complete.")
