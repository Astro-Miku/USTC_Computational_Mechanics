"""
test.py — FEM post-processing examples.

Run after E_main.py. Import from H_post_process and call directly.

=================================================================
Usage examples (uncomment to use):
=================================================================
"""

from G_get_displacement import load_result, interpolate_displacement
from H_post_process import (
    displacement_point, displacement_line, displacement_max, displacement_min,
    displacement_boundary_mean, displacement_boundary_line,
    stress_point, stress_line, stress_max, stress_min, stress_absmin,
    stress_boundary_mean, stress_boundary_line,
    principal_stress_point, principal_stress_line,
    principal_stress_max, principal_stress_min, principal_stress_absmin,
    principal_stress_boundary_mean, principal_stress_boundary_line,
    mises_point, mises_line, mises_max, mises_min, mises_absmin,
    mises_boundary_mean, mises_boundary_line,
    run_all,
)

# --- Paths to FEM results ---
GRID = 'case/grid.npz'
DISP = 'case/DisplacementResult.npz'
E = 200e9
niu = 0.3
x=0
y=0
# --- Basic displacement interpolation (original test) ---
g, U = load_result(GRID, DISP)
u, v = interpolate_displacement(g, U, x, y)
#print(u, v)

print()
print("=" * 60)
print("Post-processing")
print("=" * 60)

# =================================================================
# Point queries — prints value, returns (u,v) or (sx,sy,txy) etc.
# =================================================================

# displacement_point(x, y, GRID, DISP)
print()
displacement_point(x, y, GRID, DISP)

# stress_point(x, y, GRID, DISP, E, niu)
print()
stress_point(x, y, GRID, DISP, E, niu)

# principal_stress_point(x, y, GRID, DISP, E, niu)
print()
principal_stress_point(x, y, GRID, DISP, E, niu)

# mises_point(x, y, GRID, DISP, E, niu)
print()
mises_point(x, y, GRID, DISP, E, niu)

# =================================================================
# Line queries — saves plot to case/, returns arrays
# =================================================================

# displacement_line(x1, y1, x2, y2, GRID, DISP, n_samples=100)
print()
displacement_line(0, 0, 1, 1, GRID, DISP)

# stress_line(x1, y1, x2, y2, GRID, DISP, E, niu, n_samples=100)
print()
stress_line(0, 0, 1, 1, GRID, DISP, E, niu)

# principal_stress_line(x1, y1, x2, y2, GRID, DISP, E, niu, n_samples=100)
print()
principal_stress_line(0, 0, 1, 1, GRID, DISP, E, niu)

# mises_line(x1, y1, x2, y2, GRID, DISP, E, niu, n_samples=100)
print()
mises_line(0, 0, 1, 1, GRID, DISP, E, niu)

# =================================================================
# Global max / min — prints value and location
# =================================================================

# displacement_max(GRID, DISP)
print()
displacement_max(GRID, DISP)

# displacement_min(GRID, DISP)
print()
displacement_min(GRID, DISP)

# stress_max(GRID, DISP, E, niu)
print()
stress_max(GRID, DISP, E, niu)

# stress_min(GRID, DISP, E, niu)
print()
stress_min(GRID, DISP, E, niu)

# stress_absmin(GRID, DISP, E, niu)
print()
stress_absmin(GRID, DISP, E, niu)

# principal_stress_max(GRID, DISP, E, niu)
print()
principal_stress_max(GRID, DISP, E, niu)

# principal_stress_min(GRID, DISP, E, niu)
print()
principal_stress_min(GRID, DISP, E, niu)

# principal_stress_absmin(GRID, DISP, E, niu)
print()
principal_stress_absmin(GRID, DISP, E, niu)

# mises_max(GRID, DISP, E, niu)
print()
mises_max(GRID, DISP, E, niu)

# mises_min(GRID, DISP, E, niu)
print()
mises_min(GRID, DISP, E, niu)

# mises_absmin(GRID, DISP, E, niu)
print()
mises_absmin(GRID, DISP, E, niu)

# =================================================================
# Boundary mean (length-weighted) — boundary segment via linked list
# =================================================================

# boundary segments defined in W_give_BC:  0→112 (top edge), 32→1 (right edge)
BND_START = 0   # top edge start
BND_END   = 112  # top edge end

# displacement_boundary_mean(BND_START, BND_END, GRID, DISP)
print()
displacement_boundary_mean(BND_START, BND_END, GRID, DISP)

# stress_boundary_mean(BND_START, BND_END, GRID, DISP, E, niu)
print()
stress_boundary_mean(BND_START, BND_END, GRID, DISP, E, niu)

# principal_stress_boundary_mean(BND_START, BND_END, GRID, DISP, E, niu)
print()
principal_stress_boundary_mean(BND_START, BND_END, GRID, DISP, E, niu)

# mises_boundary_mean(BND_START, BND_END, GRID, DISP, E, niu)
print()
mises_boundary_mean(BND_START, BND_END, GRID, DISP, E, niu)

# =================================================================
# Boundary line plots — x-axis: s (arc length), x, or y; with arrows
# =================================================================

# displacement_boundary_line(BND_START, BND_END, 's', GRID, DISP)
print()
displacement_boundary_line(BND_START, BND_END, 's', GRID, DISP)

# stress_boundary_line(BND_START, BND_END, 's', GRID, DISP, E, niu)
print()
stress_boundary_line(BND_START, BND_END, 's', GRID, DISP, E, niu)

# principal_stress_boundary_line(BND_START, BND_END, 's', GRID, DISP, E, niu)
print()
principal_stress_boundary_line(BND_START, BND_END, 's', GRID, DISP, E, niu)

# mises_boundary_line(BND_START, BND_END, 's', GRID, DISP, E, niu)
print()
mises_boundary_line(BND_START, BND_END, 's', GRID, DISP, E, niu)

# =================================================================
# Convenience: run everything at once
# =================================================================
# run_all(x_point=30, y_point=1.5,
#         line_start=(0, 0), line_end=(30, 3),
#         grid_path=GRID, disp_path=DISP, E=E, niu=niu)
