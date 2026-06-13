import numpy as np
from A_module import Vertex

def load_grid_from_npz(npz_file):
    data = np.load(npz_file)
    coords = data['coords']  # N x 2
    tris = data['tris']      # M x 3
    verts = []
    for i in range(len(coords)):
        x, y = coords[i]
        verts.append(Vertex(i, float(x), float(y)))
    tris_list = []
    for i in range(len(tris)):
        a, b, c = tris[i]
        tris_list.append((int(a), int(b), int(c)))
    
    return verts, tris_list

if __name__ == "__main__":
    npz_file = "grid.npz"
    verts, tris = load_grid_from_npz(npz_file)
    