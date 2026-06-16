import numpy as np

def read_grid_txt(file_path):
    """
    Read grid.txt file, extract node coordinates and triangle elements.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # find the start of coordinate and triangle data sections
    coord_start = None
    tri_start = None
    
    for i, line in enumerate(lines):
        if '% Coordinates' in line:
            coord_start = i + 1
        elif '% Elements (triangles)' in line:
            tri_start = i + 1
            break
    
    # extract coordinate data
    coords = []
    for i in range(coord_start, tri_start - 1):
        line = lines[i].strip()
        if line:  # skip empty lines
            parts = line.split()
            if len(parts) >= 2:
                x, y = map(float, parts[:2])
                coords.append([x, y])
    
    # extract triangle data
    tris = []
    for i in range(tri_start, len(lines)):
        line = lines[i].strip()
        if line:  # skip empty lines
            parts = line.split()
            if len(parts) >= 3:
                # Note: COMSOL uses 1-based indexing, convert to 0-based
                a, b, c = map(int, parts[:3])
                tris.append([a - 1, b - 1, c - 1])  # convert to 0-based indexing
    
    return np.array(coords), np.array(tris)

def save_grid_to_npz(txt_file, npz_file):
    """
    Read txt file and save as npz file.
    """
    coords, tris = read_grid_txt(txt_file)
    
    # save to npz file
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
    txt_file = "case/grid.txt"  # path to your txt file
    npz_file = "case/grid.npz"  # output npz file path
    
    coords, tris = save_grid_to_npz(txt_file, npz_file)