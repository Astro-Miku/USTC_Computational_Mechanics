import numpy as np
import math
def N_ijk(v1,v2,v3):
    xi=v1.x
    yi=v1.y
    xj=v2.x
    yj=v2.y
    xk=v3.x
    yk=v3.y
    two_A = (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)
    Nix = (yj - yk) / two_A
    Njx = (yk - yi) / two_A
    Nkx = (yi - yj) / two_A
    Niy = (xk - xj) / two_A
    Njy = (xi - xk) / two_A
    Nky = (xj - xi) / two_A
    N = (Nix, Niy, Njx, Njy, Nkx, Nky)
    return N
def LN(N):
    Nix,Niy,Njx,Njy,Nkx,Nky=N
    B = np.array([
        [Nix, 0,   Njx, 0,   Nkx, 0],
        [0,   Niy, 0,   Njy, 0,   Nky],
        [Niy, Nix, Njy, Njx, Nky, Nkx]
    ])
    #print(N)
    #print(B)
    return B

def getD(E,niu,is_stress):
    if is_stress==0: # 0 = plane strain, 1 = plane stress
        E,niu=E/(1-niu*niu),niu/(1-niu)
    D = E/(1-niu**2) * np.array([
        [1, niu, 0],
        [niu, 1, 0],
        [0, 0, (1-niu)/2]
    ])
    return D
def normal(v1,v2):
    dx12 = v2.x - v1.x
    dy12 = v2.y - v1.y
    L12 = math.hypot(dx12, dy12)
    n12 = (dy12 / L12, -dx12 / L12)
    return n12
def edge_force(graph,tri_idx,E,niu,is_stress):
    tri=graph.triangles[tri_idx]
    v1_idx,v2_idx,v3_idx,e12_idx,e23_idx,e31_idx=tri
    v1,v2,v3=graph.vertices[v1_idx],graph.vertices[v2_idx],graph.vertices[v3_idx]
    e12,e23,e31=graph.edges[e12_idx],graph.edges[e23_idx],graph.edges[e31_idx]
    DB=np.dot(getD(E,niu,is_stress), LN(N_ijk(v1,v2,v3)))
    outnormal=[normal(v1,v2),normal(v2,v3),normal(v3,v1)]
    edge_force_trans=np.array([
        [outnormal[0][0]*e12.length, 0                         , outnormal[0][1]*e12.length],        #12-x
        [0                         , outnormal[0][1]*e12.length, outnormal[0][0]*e12.length],        #12-y
        [outnormal[1][0]*e23.length, 0                         , outnormal[1][1]*e23.length],        #23-x
        [0                         , outnormal[1][1]*e23.length, outnormal[1][0]*e23.length],        #23-y
        [outnormal[2][0]*e31.length, 0                         , outnormal[2][1]*e31.length],        #31-x
        [0                         , outnormal[2][1]*e31.length, outnormal[2][0]*e31.length],        #31-y
    ])
    return np.dot(edge_force_trans, DB)
def vertex_force(graph,tri_idx,E,niu,is_stress):
    edge_matrix=edge_force(graph,tri_idx,E,niu,is_stress)
    vertex_trans=np.array([
        [0.5, 0  , 0  , 0  , 0.5, 0  ],         #1-x
        [0  , 0.5, 0  , 0  , 0  , 0.5],         #1-y
        [0.5, 0  , 0.5, 0  , 0  , 0  ],         #2-x
        [0  , 0.5, 0  , 0.5, 0  , 0  ],         #2-y
        [0  , 0  , 0.5, 0  , 0.5, 0  ],         #3-x
        [0  , 0  , 0  , 0.5, 0  , 0.5],         #3-y
    ])
    return np.dot(vertex_trans, edge_matrix)