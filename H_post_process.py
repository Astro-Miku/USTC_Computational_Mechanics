"""
Post-processing module for 2D FEM elastic analysis.

Provides 27 functions following the naming convention:
    {target}_{mode}

target:  displacement | stress | principal_stress | mises
mode:    point | line | max | min | absmin | boundary_mean | boundary_line

Calling convention:
    target_mode(location, grid_path, disp_path, E=..., niu=...)

    - point:         target_point(x, y, grid_path, disp_path, ...)
    - line:          target_line(x1, y1, x2, y2, grid_path, disp_path, ...)
                     -> saves a plot to case/ and returns data arrays
    - max:           target_max(grid_path, disp_path, ...)
    - min:           target_min(grid_path, disp_path, ...)
    - absmin:        target_absmin(grid_path, disp_path, ...)
    - boundary_mean: target_boundary_mean(start, end, grid_path, disp_path, ...)
                     -> length-weighted average along boundary linked list
    - boundary_line: target_boundary_line(start, end, x_axis, grid_path, ...)
                     -> plot along boundary, x-axis: s | x | y, with arrows

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


def _boundary_mean(g, U, E, niu, start, end, link_dict, value_fn, n_pts=3):
    """
    Length-weighted mean of a quantity along a boundary segment.

    Traverses the boundary linked list from start → end.
    n_pts=1 : midpoint rule (1 sample per edge)
    n_pts=3 : Simpson rule (endpoints + midpoint, weights 1:4:1)

    value_fn: (g, U, x, y, E, niu) -> float | tuple | None
    Returns: scalar or tuple of component means (None if no valid data)
    """
    # collect sample points with integration weights along the chain
    samples = []   # (x, y, weight)
    p = start
    total_length = 0.0
    while p != end:
        q = link_dict.get(p)
        if q is None:
            break
        vp = g.vertices[p]
        vq = g.vertices[q]
        length = np.hypot(vp.x - vq.x, vp.y - vq.y)

        if n_pts == 3:
            mid_x = (vp.x + vq.x) * 0.5
            mid_y = (vp.y + vq.y) * 0.5
            samples.append((vp.x, vp.y, length / 6.0))
            samples.append((mid_x, mid_y, 4.0 * length / 6.0))
            samples.append((vq.x, vq.y, length / 6.0))
        else:  # midpoint only
            mid_x = (vp.x + vq.x) * 0.5
            mid_y = (vp.y + vq.y) * 0.5
            samples.append((mid_x, mid_y, length))

        total_length += length
        p = q

    if total_length == 0.0:
        return None

    # accumulate weighted sum
    n_comps = None
    weighted_sum = None
    for x, y, w in samples:
        v = value_fn(g, U, x, y, E, niu)
        if v is None:
            continue
        if not isinstance(v, (tuple, list, np.ndarray)):
            v = (v,)
        if n_comps is None:
            n_comps = len(v)
            weighted_sum = [0.0] * n_comps
        for i in range(n_comps):
            weighted_sum[i] += v[i] * w

    if weighted_sum is None:
        return None

    means = tuple(s / total_length for s in weighted_sum)
    return means[0] if n_comps == 1 else means


def _load_link_dict(link_path='case/boun_link.npz'):
    """Load boundary linked-list dictionary from npz file."""
    data = np.load(link_path)
    return {u: v for u, v in data['link']}


def _boundary_sample(g, U, E, niu, start, end, link_dict, value_fn):
    """
    Sample a quantity at every boundary node along the linked-list chain.

    Returns: (xs, ys, ss, data_arrays)
      xs, ys: coordinates of sample points (in traversal order)
      ss:     cumulative arc length from start
      data_arrays: list of np.ndarray, one per component
    """
    # collect node indices in traversal order
    node_ids = [start]
    p = start
    while p != end:
        q = link_dict.get(p)
        if q is None:
            break
        node_ids.append(q)
        p = q

    # sample at each node
    xs, ys, ss = [], [], []
    val_lists = None
    n_comps = None
    cum_s = 0.0
    px = py = None

    for k, nid in enumerate(node_ids):
        v = g.vertices[nid]
        if k > 0:
            cum_s += np.hypot(v.x - px, v.y - py)
        xs.append(v.x)
        ys.append(v.y)
        ss.append(cum_s)
        px, py = v.x, v.y

        val = value_fn(g, U, v.x, v.y, E, niu)
        if val is not None:
            if not isinstance(val, (tuple, list, np.ndarray)):
                val = (val,)
            if n_comps is None:
                n_comps = len(val)
                val_lists = [[] for _ in range(n_comps)]
            for j in range(n_comps):
                val_lists[j].append(val[j])
        else:
            if val_lists is not None:
                for lst in val_lists:
                    lst.append(np.nan)

    if val_lists is None:
        return np.array(xs), np.array(ys), np.array(ss), []
    return (np.array(xs), np.array(ys), np.array(ss),
            [np.array(lst) for lst in val_lists])


def _add_boundary_arrows(ax, xs, ys, ss, n_arrows=12, color='black'):
    """Add direction arrows at regular arc-length intervals along a polyline."""
    if len(xs) < 2 or n_arrows <= 0:
        return
    total_s = ss[-1]
    if total_s <= 0:
        return
    for s_target in np.linspace(total_s * 0.05, total_s * 0.95, n_arrows):
        # locate segment containing s_target
        for i in range(len(ss) - 1):
            if ss[i] <= s_target <= ss[i + 1]:
                seg_len = ss[i + 1] - ss[i]
                if seg_len <= 0:
                    break
                t = (s_target - ss[i]) / seg_len
                x = xs[i] + t * (xs[i + 1] - xs[i])
                y = ys[i] + t * (ys[i + 1] - ys[i])
                dx = xs[i + 1] - xs[i]
                dy = ys[i + 1] - ys[i]
                length = np.hypot(dx, dy)
                if length > 0:
                    dx /= length; dy /= length
                    ax.annotate('', xy=(x + dx * 0.02, y + dy * 0.02),
                                xytext=(x - dx * 0.02, y - dy * 0.02),
                                arrowprops=dict(arrowstyle='->', color=color,
                                                lw=1.2, alpha=0.7))
                break


def _boundary_line_plot(start, end, xs, ys, ss,
                        data_arrays, labels, title, filename, x_axis):
    """Save a boundary line-plot with arrows. x_axis: 's' | 'x' | 'y'."""
    # ============ X-AXIS SWITCH ============
    #  x_data / x_label  →  horizontal axis of the line plot.
    #  To customise (e.g. angle = y / R):
    #     x_data = ys / R
    #     x_label = 'angle'
    # ========================================
    if x_axis == 's':
        x_data = ss
        x_label = 'arc length s'
    elif x_axis == 'x':
        x_data = xs
        x_label = 'x'
    else:
        x_data = ys
        x_label = 'y'

    n = len(data_arrays)
    _, axes = plt.subplots(n, 1, figsize=(10, 2.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, (arr, lbl) in enumerate(zip(data_arrays, labels)):
        # connect adjacent sample points in traversal order
        axes[i].plot(x_data, arr, 'o-', linewidth=1.2, markersize=3, label=lbl)
        axes[i].set_ylabel(lbl)
        axes[i].grid(True, alpha=0.3)
        # add direction arrows at midpoints
        _add_boundary_arrows(axes[i], x_data, arr, ss if x_axis == 's'
                             else np.arange(len(x_data)))

    axes[-1].set_xlabel(x_label)
    axes[0].set_title(title + f'  (boundary {start}->{end})')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {filename}")


def _boundary_sample_edge(g, U, E, niu, start, end, link_dict, value_fn):
    """
    Sample at each boundary edge midpoint — correct for stress-type
    quantities that are constant per element but jump at nodes.

    Returns: (s_starts, s_ends, x0, x1, y0, y1, data_arrays)
      s_starts / s_ends: arc-length bounds of each edge
      x0 / x1: x-coordinate of edge start / end node
      y0 / y1: y-coordinate of edge start / end node
      data_arrays: list of np.ndarray, one value per edge
    """
    s_starts, s_ends = [], []
    x0, x1, y0, y1 = [], [], [], []
    val_lists = None
    n_comps = None
    cum_s = 0.0
    p = start

    while p != end:
        q = link_dict.get(p)
        if q is None:
            break
        vp = g.vertices[p]
        vq = g.vertices[q]
        length = np.hypot(vp.x - vq.x, vp.y - vq.y)
        mid_x = (vp.x + vq.x) * 0.5
        mid_y = (vp.y + vq.y) * 0.5

        s_starts.append(cum_s)
        cum_s += length
        s_ends.append(cum_s)
        x0.append(vp.x)
        x1.append(vq.x)
        y0.append(vp.y)
        y1.append(vq.y)

        val = value_fn(g, U, mid_x, mid_y, E, niu)
        if val is not None:
            if not isinstance(val, (tuple, list, np.ndarray)):
                val = (val,)
            if n_comps is None:
                n_comps = len(val)
                val_lists = [[] for _ in range(n_comps)]
            for j in range(n_comps):
                val_lists[j].append(val[j])
        else:
            if val_lists is not None:
                for lst in val_lists:
                    lst.append(np.nan)

        p = q

    if val_lists is None:
        return (np.array([]), np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]), [])
    return (np.array(s_starts), np.array(s_ends),
            np.array(x0), np.array(x1), np.array(y0), np.array(y1),
            [np.array(lst) for lst in val_lists])


def _boundary_edge_plot(start, end, s_starts, s_ends, x0, x1, y0, y1,
                        data_arrays, labels, title, filename, x_axis):
    """
    Boundary step plot — constant value across each edge, jump at node.
    x_axis: 's' (arc length) | 'x' | 'y'
    """
    n_edges = len(s_starts)
    if n_edges == 0:
        return

    # ============ X-AXIS SWITCH ============
    #  x_left / x_right → step-segment left & right endpoints on the x-axis.
    #  arrow_span       → direction of traversal arrows (+→right, -→left).
    #  x_label          → axis title.
    #
    #  To customise (e.g. angle = y_mid / R, arc-length weighted):
    #     x_left  = y0 / R          # start-node angle
    #     x_right = y1 / R          # end-node angle
    #     x_label = 'angle'
    #     arrow_span = x_right - x_left
    #
    #  To customise (e.g. plot vs edge index):
    #     x_left  = np.arange(n_edges)
    #     x_right = np.arange(n_edges) + 1
    #     x_label = 'edge index'
    #     arrow_span = np.ones(n_edges)
    # ========================================
    if x_axis == 's':
        x_left = s_starts
        x_right = s_ends
        x_label = 'arc length s'
        arrow_span = s_ends - s_starts
    elif x_axis == 'x':
        x_left = x0
        x_right = x1
        x_label = 'x'
        arrow_span = x1 - x0
    else:  # 'y'
        x_left = y0
        x_right = y1
        x_label = 'y'
        arrow_span = y1 - y0

    # interleave: [L0, R0, L1, R1, ...] for step drawing
    x_step = np.empty(2 * n_edges)
    for i in range(n_edges):
        x_step[2 * i] = x_left[i]
        x_step[2 * i + 1] = x_right[i]

    n = len(data_arrays)
    _, axes = plt.subplots(n, 1, figsize=(10, 2.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, (vals, lbl) in enumerate(zip(data_arrays, labels)):
        y_step = np.empty(2 * n_edges)
        for j in range(n_edges):
            y_step[2 * j] = vals[j]
            y_step[2 * j + 1] = vals[j]

        axes[i].plot(x_step, y_step, '-', linewidth=1.2, label=lbl)
        axes[i].set_ylabel(lbl)
        axes[i].grid(True, alpha=0.3)

        # arrows at midpoints, pointing along traversal direction
        if n_edges >= 2:
            stp = max(1, n_edges // 10)
            for j in range(0, n_edges, stp):
                mx = 0.5 * (x_left[j] + x_right[j])
                dx = arrow_span[j]
                if dx >= 0:
                    axes[i].annotate('', xy=(mx + 0.005, vals[j]),
                                     xytext=(mx - 0.005, vals[j]),
                                     arrowprops=dict(arrowstyle='->', color='black',
                                                     lw=1.2, alpha=0.6))
                else:
                    axes[i].annotate('', xy=(mx - 0.005, vals[j]),
                                     xytext=(mx + 0.005, vals[j]),
                                     arrowprops=dict(arrowstyle='->', color='black',
                                                     lw=1.2, alpha=0.6))

    axes[-1].set_xlabel(x_label)
    axes[0].set_title(title + f'  (boundary {start}->{end})')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {filename}")


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
    mag = np.sqrt(u**2 + v**2)
    print(f"Displacement at ({x}, {y}): u = {u:.6e}, v = {v:.6e}, |u| = {mag:.6e}")
    return u, v, mag


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
# Public API — boundary mean (length-weighted)
# ============================================================

def displacement_boundary_mean(start, end,
                                grid_path='case/grid.npz',
                                disp_path='case/DisplacementResult.npz',
                                link_path='case/boun_link.npz'):
    """Print length-weighted mean displacement (u, v, |u|) along a boundary segment."""
    g, U = _load(grid_path, disp_path)
    link_dict = _load_link_dict(link_path)

    def _uv(g, U, x, y, E, niu):
        return interpolate_displacement(g, U, x, y)

    def _mag(g, U, x, y, E, niu):
        uv = interpolate_displacement(g, U, x, y)
        if uv is None:
            return None
        return np.sqrt(uv[0]**2 + uv[1]**2)

    result = _boundary_mean(g, U, 200e9, 0.3, start, end, link_dict, _uv)
    mag_mean = _boundary_mean(g, U, 200e9, 0.3, start, end, link_dict, _mag)
    if result is None:
        print(f"No valid displacement data on boundary ({start}->{end}).")
        return None, None, None
    u_mean, v_mean = result
    print(f"Boundary mean displacement ({start}->{end}):"
          f" u_avg = {u_mean:.6e}, v_avg = {v_mean:.6e},"
          f" |u|_avg = {mag_mean:.6e}")
    return u_mean, v_mean, mag_mean


def stress_boundary_mean(start, end,
                          grid_path='case/grid.npz',
                          disp_path='case/DisplacementResult.npz',
                          E=200e9, niu=0.3,
                          link_path='case/boun_link.npz'):
    """Print length-weighted mean stress (sigma_x, sigma_y, tau_xy) along a boundary segment."""
    g, U = _load(grid_path, disp_path)
    link_dict = _load_link_dict(link_path)

    result = _boundary_mean(g, U, E, niu, start, end, link_dict, _stress_at_point)
    if result is None:
        print(f"No valid stress data on boundary ({start}->{end}).")
        return None, None, None
    sx_mean, sy_mean, txy_mean = result
    print(f"Boundary mean stress ({start}->{end}):"
          f" sigma_x_avg = {sx_mean:.6e}, sigma_y_avg = {sy_mean:.6e},"
          f" tau_xy_avg = {txy_mean:.6e}")
    return sx_mean, sy_mean, txy_mean


def principal_stress_boundary_mean(start, end,
                                    grid_path='case/grid.npz',
                                    disp_path='case/DisplacementResult.npz',
                                    E=200e9, niu=0.3,
                                    link_path='case/boun_link.npz'):
    """Print length-weighted mean principal stresses (sigma_1, sigma_2) along a boundary segment."""
    g, U = _load(grid_path, disp_path)
    link_dict = _load_link_dict(link_path)

    result = _boundary_mean(g, U, E, niu, start, end, link_dict, _principal_at_point)
    if result is None:
        print(f"No valid principal stress data on boundary ({start}->{end}).")
        return None, None
    s1_mean, s2_mean = result
    print(f"Boundary mean principal stress ({start}->{end}):"
          f" sigma_1_avg = {s1_mean:.6e}, sigma_2_avg = {s2_mean:.6e}")
    return s1_mean, s2_mean


def mises_boundary_mean(start, end,
                         grid_path='case/grid.npz',
                         disp_path='case/DisplacementResult.npz',
                         E=200e9, niu=0.3,
                         link_path='case/boun_link.npz'):
    """Print length-weighted mean von Mises stress along a boundary segment."""
    g, U = _load(grid_path, disp_path)
    link_dict = _load_link_dict(link_path)

    result = _boundary_mean(g, U, E, niu, start, end, link_dict, _mises_at_point)
    if result is None:
        print(f"No valid Mises data on boundary ({start}->{end}).")
        return None
    print(f"Boundary mean Mises ({start}->{end}):"
          f" Mises_avg = {result:.6e}")
    return result


# ============================================================
# Public API — boundary line plot (traversal order, with arrows)
# ============================================================

def displacement_boundary_line(start, end, x_axis='s',
                                grid_path='case/grid.npz',
                                disp_path='case/DisplacementResult.npz',
                                link_path='case/boun_link.npz'):
    """
    Plot displacement components along a boundary segment.
    x_axis: 's' (arc length) | 'x' | 'y'
    """
    g, U = _load(grid_path, disp_path)
    link_dict = _load_link_dict(link_path)

    def _uv(g, U, x, y, E, niu):
        uv = interpolate_displacement(g, U, x, y)
        if uv is None:
            return None
        return uv[0], uv[1], np.sqrt(uv[0]**2 + uv[1]**2)

    xs, ys, ss, arrays = _boundary_sample(g, U, 200e9, 0.3,
                                           start, end, link_dict, _uv)
    if not arrays or len(arrays) < 3:
        print(f"No valid data on boundary ({start}->{end}).")
        return None
    u_arr, v_arr, mag_arr = arrays
    fname = f'case/displacement_boundary_{start}_{end}_{x_axis}.png'
    _boundary_line_plot(start, end, xs, ys, ss,
                        [u_arr, v_arr, mag_arr],
                        ['u', 'v', '|u|'],
                        'Displacement along boundary', fname, x_axis)
    return u_arr, v_arr, mag_arr


def stress_boundary_line(start, end, x_axis='s',
                          grid_path='case/grid.npz',
                          disp_path='case/DisplacementResult.npz',
                          E=200e9, niu=0.3,
                          link_path='case/boun_link.npz'):
    """
    Plot stress components along a boundary segment (step plot: constant per edge).
    x_axis: 's' (arc length) | 'x' | 'y'
    """
    g, U = _load(grid_path, disp_path)
    link_dict = _load_link_dict(link_path)

    s_starts, s_ends, x0, x1, y0, y1, arrays = _boundary_sample_edge(
        g, U, E, niu, start, end, link_dict, _stress_at_point)
    if not arrays or len(arrays) < 3:
        print(f"No valid data on boundary ({start}->{end}).")
        return None
    sx_arr, sy_arr, txy_arr = arrays
    fname = f'case/stress_boundary_{start}_{end}_{x_axis}.png'
    _boundary_edge_plot(start, end, s_starts, s_ends, x0, x1, y0, y1,
                        [sx_arr, sy_arr, txy_arr],
                        ['sigma_x', 'sigma_y', 'tau_xy'],
                        'Stress along boundary', fname, x_axis)
    return sx_arr, sy_arr, txy_arr


def principal_stress_boundary_line(start, end, x_axis='s',
                                    grid_path='case/grid.npz',
                                    disp_path='case/DisplacementResult.npz',
                                    E=200e9, niu=0.3,
                                    link_path='case/boun_link.npz'):
    """
    Plot principal stresses along a boundary segment (step plot: constant per edge).
    x_axis: 's' (arc length) | 'x' | 'y'
    """
    g, U = _load(grid_path, disp_path)
    link_dict = _load_link_dict(link_path)

    s_starts, s_ends, x0, x1, y0, y1, arrays = _boundary_sample_edge(
        g, U, E, niu, start, end, link_dict, _principal_at_point)
    if not arrays or len(arrays) < 2:
        print(f"No valid data on boundary ({start}->{end}).")
        return None
    s1_arr, s2_arr = arrays
    fname = f'case/principal_stress_boundary_{start}_{end}_{x_axis}.png'
    _boundary_edge_plot(start, end, s_starts, s_ends, x0, x1, y0, y1,
                        [s1_arr, s2_arr],
                        ['sigma_1', 'sigma_2'],
                        'Principal stress along boundary', fname, x_axis)
    return s1_arr, s2_arr


def mises_boundary_line(start, end, x_axis='s',
                         grid_path='case/grid.npz',
                         disp_path='case/DisplacementResult.npz',
                         E=200e9, niu=0.3,
                         link_path='case/boun_link.npz'):
    """
    Plot von Mises stress along a boundary segment (step plot: constant per edge).
    x_axis: 's' (arc length) | 'x' | 'y'
    """
    g, U = _load(grid_path, disp_path)
    link_dict = _load_link_dict(link_path)

    s_starts, s_ends, x0, x1, y0, y1, arrays = _boundary_sample_edge(
        g, U, E, niu, start, end, link_dict, _mises_at_point)
    if not arrays or len(arrays) < 1:
        print(f"No valid data on boundary ({start}->{end}).")
        return None
    mises_arr = arrays[0]
    fname = f'case/mises_boundary_{start}_{end}_{x_axis}.png'
    _boundary_edge_plot(start, end, s_starts, s_ends, x0, x1, y0, y1,
                        [mises_arr],
                        ['Von Mises'],
                        'Von Mises along boundary', fname, x_axis)
    return mises_arr


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
