import numpy as np
from A_module import Vertex, Edge, Graph
from F_find_tri import find_triangle_of_point
from Y_get_info import load_grid_from_npz
def load_result(graph_path,U_path):
    verts, tris = load_grid_from_npz(graph_path)
    data = np.load(U_path)
    U=data['U']
    g = Graph(verts)
    g.build(tris)
    return g,U
def interpolate_displacement(graph, U, px, py):
    tri = find_triangle_of_point(graph, px, py)
    if tri is None:
        return None

    i1, i2, i3, *_ = tri
    v1, v2, v3 = graph.vertices[i1], graph.vertices[i2], graph.vertices[i3]

    x1, y1 = v1.x, v1.y
    x2, y2 = v2.x, v2.y
    x3, y3 = v3.x, v3.y

    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    w1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
    w2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
    w3 = 1 - w1 - w2

    u = w1 * U[2 * i1] + w2 * U[2 * i2] + w3 * U[2 * i3]
    v = w1 * U[2 * i1 + 1] + w2 * U[2 * i2 + 1] + w3 * U[2 * i3 + 1]

    return u, v
