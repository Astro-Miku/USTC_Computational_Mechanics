import numpy as np


def read_mphtxt(file_path):
    """
    Read COMSOL .mphtxt file, extract node coordinates and triangle elements.
    mphtxt uses 0-based indexing natively, so no conversion is needed.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # --- helper: strip inline comments and whitespace ---
    def clean_line(line):
        if '#' in line:
            line = line.split('#')[0]
        return line.strip()

    # --- parse number of mesh points ---
    n_points = None
    for line in lines:
        if '# number of mesh points' in line:
            n_points = int(clean_line(line))
            break

    # --- locate coordinate section ---
    coord_start = None
    for i, line in enumerate(lines):
        if '# Mesh point coordinates' in line:
            coord_start = i + 1
            break

    # --- extract coordinates ---
    coords = []
    j = coord_start
    while len(coords) < n_points and j < len(lines):
        line = clean_line(lines[j])
        j += 1
        if line:
            parts = line.split()
            if len(parts) >= 2:
                x, y = map(float, parts[:2])
                coords.append([x, y])

    # --- locate triangle type section (type name contains 'tri') ---
    tri_section_idx = None
    for i, line in enumerate(lines):
        s = line.strip()
        if not s.startswith('#') and 'tri' in s and '# type name' in s:
            tri_section_idx = i
            break

    if tri_section_idx is None:
        raise ValueError("No 'tri' element type found in the mphtxt file.")

    # --- parse number of triangle elements ---
    n_tris = None
    for i in range(tri_section_idx, len(lines)):
        if '# number of elements' in lines[i]:
            n_tris = int(clean_line(lines[i]))
            break

    # --- locate triangle element data ---
    tri_data_start = None
    for i in range(tri_section_idx, len(lines)):
        if '# Elements' in lines[i]:
            tri_data_start = i + 1
            break

    # --- extract triangle elements (already 0-based in mphtxt) ---
    tris = []
    j = tri_data_start
    while len(tris) < n_tris and j < len(lines):
        line = clean_line(lines[j])
        j += 1
        if line:
            parts = line.split()
            if len(parts) >= 3:
                a, b, c = map(int, parts[:3])
                tris.append([a, b, c])

    return np.array(coords), np.array(tris)


def save_mphtxt_to_npz(mphtxt_file, npz_file):
    """
    Read COMSOL .mphtxt file and save as .npz file.
    Output format is identical to save_grid_to_npz in Z_readtxt.py.
    """
    coords, tris = read_mphtxt(mphtxt_file)

    np.savez(npz_file,
             coords=coords,
             tris=tris,
             n_nodes=len(coords),
             n_tris=len(tris))

    print(f"Saved to {npz_file}")
    print(f"Number of nodes: {len(coords)}")
    print(f"Number of triangles: {len(tris)}")

    return coords, tris


# Usage example
if __name__ == "__main__":
    mphtxt_file = "case/grid.mphtxt"   # path to your COMSOL mphtxt file
    npz_file = "case/grid.npz"  # output npz file path

    coords, tris = save_mphtxt_to_npz(mphtxt_file, npz_file)
