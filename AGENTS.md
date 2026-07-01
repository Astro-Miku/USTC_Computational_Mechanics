# AGENTS.md — 2D FEM Elastic Analysis

A **linear-elastic 2D Finite Element solver** using constant-strain triangle (CST)
elements with COMSOL-generated meshes. The workflow goes: mesh → assembly → solve →
post-process (displacement / stress / principal stress / von Mises).

---

## Build / Test / Lint

No build system, no test framework. Pure Python 3 with `numpy`, `scipy`, `matplotlib`.

| Action | Command |
|--------|---------|
| Install deps | `pip install numpy scipy matplotlib` |
| Full solve | `cd case && python ../W_give_BC.py` (BC setup), then `python ../E_main.py` |
| Post-process | `python test.py` |
| Quick check | `python -c "from Y_get_info import load_grid_from_npz; verts, tris = load_grid_from_npz('case/grid.npz'); print(f'{len(verts)} nodes, {len(tris)} triangles')"` |

---

## Architecture

### Module Dependency Order (A → H)

```
Z_readmphtxt.py / Z_readtxt.py  ──►  Y_get_info.py  ──►  A_module.py
                                                                │
                     ┌──────────────────────────────────────────┘
                     ▼
              B_calcu_metrix.py   (element stiffness)
                     │
                     ▼
              C_build_whole.py    (global assembly via CSR)
                     │
                     ▼
              D_give_boundary.py  (apply Dirichlet BCs, build RHS)
                     │
                     ▼
              E_main.py           (solve K·U = F → save displacement)
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
  F_find_tri.py            G_get_displacement.py
  (point-in-triangle)      (interpolate at arbitrary point)
         │                       │
         └───────────┬───────────┘
                     ▼
              H_post_process.py   (27 public functions)

  Support files:
    X_GeometryVisual.py  — extract boundary loops from mesh, save boun_link.npz
    W_give_BC.py         — define displacement/force BC using linked-list traversal
    test.py              — example post-processing script
    analysis.py          — beam theory cross-check (not part of main pipeline)
```

### Data Flow

1. **Mesh** — COMSOL `.mphtxt` → `Z_readmphtxt.py` → `case/grid.npz` (`coords`, `tris`)
2. **Boundary topology** — `X_GeometryVisual.py` → `case/boun_link.npz` (linked-list edges)
3. **BC values** — `W_give_BC.py` → `case/Boun_Cond.npz` (`Kind`: 1=disp, 2=force; `Boundary`: values)
4. **Solve** — `E_main.py`:
   - Loads mesh → `Graph` (A_module)
   - Computes element stiffness (B) → assembles global K (C) → applies BCs (D)
   - Solves `spsolve(K, F)` → saves `case/DisplacementResult.npz`
5. **Post-process** — `H_post_process.py` reads grid + displacement, interpolates via `F_find_tri.py`

### Key Design Choices

- **Constraint application**: Penalty method — rows of K for Dirichlet DOFs are zeroed and set to 1 on diagonal; F entries carry the prescribed value. The `Kind` matrix (1=displacement, 2=force) from `Boun_Cond.npz` drives this.
- **Global matrix**: Assembled as COO then converted to CSR; each 2×2 block corresponds to DOF pair `(2i, 2i+1)` for vertex i.
- **Plane stress** by default (`is_stress=1` in `B_calcu_metrix.getD`). Pass `is_stress=0` for plane strain.
- **Boundary traversal**: Boundary edges are stored as a linked list (vertex index → next vertex) in `boun_link.npz`. Functions like `BC_def` and `Value_def` walk this list.

---

## Key Files & Directories

### Top-level modules

| File | Purpose |
|------|---------|
| `A_module.py` | `Vertex`, `Edge`, `Graph` classes; graph-building from triangles |
| `B_calcu_metrix.py` | CST shape functions, B matrix, D matrix, element stiffness via `vertex_force()` |
| `C_build_whole.py` | Assembles element matrices into sparse global K (`calcu_cell` + `build_whole`) |
| `D_give_boundary.py` | Applies displacement constraints; returns `(K, F)` for solver |
| `E_main.py` | **Main entry** — loads mesh, builds system, solves, saves result + plots |
| `F_find_tri.py` | Point-in-triangle search: nearest-vertex → nearest-edge → barycentric test |
| `G_get_displacement.py` | `load_result()` + `interpolate_displacement()` at arbitrary (x, y) |
| `H_post_process.py` | **27 functions** — see "Post-processing API" below |
| `W_give_BC.py` | `BC` class for defining BCs along boundary linked-list; also has visualization |
| `X_GeometryVisual.py` | Extracts boundary edge loops, visualizes and saves `boun_link.npz` |
| `Y_get_info.py` | Loader: reads `grid.npz` → list of `Vertex` + triangle indices |
| `Z_readmphtxt.py` | Parses COMSOL `.mphtxt` → `.npz` |
| `Z_readtxt.py` | Parses generic `.txt` grid format → `.npz` |
| `test.py` | Example caller for all post-processing functions |
| `analysis.py` | Beam theory reference solution (validation use only) |

### `case/` directory

| File | Description |
|------|-------------|
| `grid.mphtxt` | COMSOL mesh (158 nodes, 256 triangles) |
| `grid.npz` | Converted mesh: `coords` (N×2), `tris` (M×3), `n_nodes`, `n_tris` |
| `boun_link.npz` | Boundary linked list: `link` (edge pairs), `head`, `nlink` |
| `Boun_Cond.npz` | BC values: `Kind` (N×2), `Boundary` (N×2) |
| `DisplacementResult.npz` | Solver output: `U` (2N,) |
| `*.png` | Visualization outputs from various runs |

---

## Coding Conventions

- **Module naming**: Prefix letters A–Z roughly indicate execution order. Internal helpers use `_` prefix.
- **Public API naming** (H_post_process): `{target}_{mode}` — e.g. `displacement_point`, `stress_line`, `mises_boundary_mean`.
- **Vertex indexing**: 0-based, matching the COMSOL .mphtxt native index.
- **DOF mapping**: Vertex `i` → DOFs `2i` (x) and `2i+1` (y).
- **Material**: Young's modulus `E`, Poisson ratio `niu` (not `nu`). Passed as floats everywhere.
- **Error handling**: Functions return `None` for out-of-mesh query points rather than raising exceptions.
- **No tests**: No test framework is present. Verification is manual (visual plots, beam cross-check in `analysis.py`).

---

## Post-processing API (H_post_process.py)

All functions follow these conventions:

| Function family | Parameters (first positional) | Returns |
|-----------------|-------------------------------|---------|
| `*_point(x, y, ...)` | x, y | Prints + returns value |
| `*_line(x1, y1, x2, y2, ...)` | line endpoints | Saves plot to `case/`, returns arrays |
| `*_max / *_min / *_absmin(...)` | — | Prints extremum + location |
| `*_boundary_mean(start, end, ...)` | boundary node indices | Prints length-weighted mean |
| `*_boundary_line(start, end, x_axis, ...)` | node indices, axis (`s`/`x`/`y`) | Saves plot to `case/` |

Default kwargs: `grid_path='case/grid.npz'`, `disp_path='case/DisplacementResult.npz'`,
`E=200e9`, `niu=0.3`.

### Available function list

- **Displacement**: `displacement_point`, `displacement_line`, `displacement_max`, `displacement_min`, `displacement_boundary_mean`, `displacement_boundary_line`
- **Stress**: `stress_point`, `stress_line`, `stress_max`, `stress_min`, `stress_absmin`, `stress_boundary_mean`, `stress_boundary_line`
- **Principal stress**: `principal_stress_point`, `principal_stress_line`, `principal_stress_max`, `principal_stress_min`, `principal_stress_absmin`, `principal_stress_boundary_mean`, `principal_stress_boundary_line`
- **Von Mises**: `mises_point`, `mises_line`, `mises_max`, `mises_min`, `mises_absmin`, `mises_boundary_mean`, `mises_boundary_line`
- **Batch**: `run_all(...)`

---

## Git Workflow

- Branch: `main` only (single-developer project).
- Commit messages: short, single-line English (e.g. `repair`, `after`, `测试数据`).
- No PR workflow, no CI. The `.gitignore` excludes `.codewhale/` and `.deepseek/` directories.

---

## Tips for AI Agents

1. **Execution order matters**: Always run `W_give_BC.py` **before** `E_main.py` — the boundary condition file must exist.
2. **The mesh is small** (158 nodes, 256 triangles). Solves are near-instant (`spsolve`).
3. **BC definition is manual** in `W_give_BC.py`. To modify a case, edit the `BC_def` and `Value_def` calls near the `# USER` section.
4. **`boun_link.npz`** defines the boundary topology. If you change the mesh, regenerate it via `X_GeometryVisual.py`.
5. **Plane stress vs. strain**: controlled by `is_stress` in `B_calcu_metrix.getD()`. The main pipeline uses `is_stress=1` (plane stress) everywhere.
6. **Out-of-mesh queries** return `None` silently — no error is raised. Always check for `None` when calling `interpolate_displacement`.
7. **`analysis.py`** is a beam-theory reference solution for validation; not part of the main pipeline.
8. **No `__init__.py`** — modules are imported directly (e.g. `from A_module import Vertex`). The workspace root acts as the package directory.

---

## CI/CD

None. No automated testing, linting, or formatting is configured.
