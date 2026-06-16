from Y_get_info import load_grid_from_npz
from A_module import Vertex, Edge, Graph
import numpy as np
import math

class BC:
    def __init__(self, verts):
        ids = {vert.idx for vert in verts}
        self.len = len(ids)
        self.kind = {nid: [2, 2] for nid in ids}        # 1=displacement, 2=force
        self.boundary = {nid: [0.0, 0.0] for nid in ids}

    def BC_def(self, start, end, constrain_x=1, constrain_y=1):
        if start == end:
            p = start
            self.kind[p] = [constrain_x, constrain_y]
            return
        p = start
        self.kind[p] = [constrain_x, constrain_y]
        while p != end:
            p = link_dict[p]
            self.kind[p] = [constrain_x, constrain_y]

    def Value_def(self, start, end, graph, funct):
        if start == end:
            p = start
            v = graph.vertices[p]
            self.boundary[p] = funct(v.x, v.y)
            return
        force_accum = {nid: [0.0, 0.0] for nid in self.kind.keys()}
        p = start
        while True:
            q = link_dict.get(p)
            if q is None:
                break
            vp = graph.vertices[p]
            vq = graph.vertices[q]
            kx, ky = self.kind[p]
            if kx == 1 or ky == 1:
                val_x, val_y = funct(vp.x, vp.y)
                if kx == 1:  # x-direction has displacement BC
                    self.boundary[p][0] = val_x
                if ky == 1:  # y-direction has displacement BC
                    self.boundary[p][1] = val_y
            if kx == 2 or ky == 2:
                mid_x = (vp.x + vq.x) / 2.0
                mid_y = (vp.y + vq.y) / 2.0
                fx, fy = funct(mid_x, mid_y)
                dx = vp.x - vq.x
                dy = vp.y - vq.y
                length = math.hypot(dx, dy)
                Fx = fx * length
                Fy = fy * length
                if kx == 2:  
                    force_accum[p][0] += Fx / 2.0
                    force_accum[q][0] += Fx / 2.0
                if ky == 2:  
                    force_accum[p][1] += Fy / 2.0
                    force_accum[q][1] += Fy / 2.0

            if q == end:
                kx_end, ky_end = self.kind[q]
                if kx_end == 1 or ky_end == 1:
                    val_x, val_y = funct(vq.x, vq.y)
                    if kx_end == 1:
                        self.boundary[q][0] = val_x
                    if ky_end == 1:
                        self.boundary[q][1] = val_y
                break
            p = q
        for nid in force_accum:
            kx, ky = self.kind[nid]
            if kx == 2:
                self.boundary[nid][0] = force_accum[nid][0]
            if ky == 2:
                self.boundary[nid][1] = force_accum[nid][1]

    def save(self, name):
        indices = sorted(self.kind.keys())
        kind_arr = np.zeros((self.len, 2))
        bound_arr = np.zeros((self.len, 2))

        for i, idx in enumerate(indices):
            kind_arr[i] = self.kind[idx]
            bound_arr[i] = self.boundary[idx]
        np.savez(name,
                 Kind=kind_arr,
                 Boundary=bound_arr)



"""

Example:

boundary.BC_def(start,end)
def funct(x,y):
    v1=x+y
    v2=x-y
    return [v1,v2]
boundary.Value_def(start,end,g,funct)
boundary.save(name)

""" 

"""
verts = [
        Vertex(0, 0.0, 0.0),
        Vertex(1, 1.0, 0.0),
        Vertex(2, 2.0, 0.0),
        Vertex(3, 0.5, 0.5),
        Vertex(4, 1.5, 0.5),
        Vertex(5, 0.0, 1.0),
        Vertex(6, 1.0, 1.0),
        Vertex(7, 2.0, 1.0),
    ]
tris = [(0,1,3),(1,6,3),(5,6,3),(5,0,3),(1,2,4),(1,6,4),(7,6,4),(7,2,4)]
"""
npz_file = "case/grid.npz"
verts, tris = load_grid_from_npz(npz_file)
g = Graph(verts)
g.build(tris)
data = np.load('case/boun_link.npz')
link_dict = {u: v for u, v in data['link']}

boundary = BC(verts)

# USER
#==================================================================================================
boundary.BC_def(32, 1,constrain_x=1,constrain_y=1)
boundary.BC_def(157, 143,constrain_x=1,constrain_y=1)
#boundary.BC_def(0, 0,constrain_x=1,constrain_y=1) 
def funct1(x, y):
    return [0, -500]
def funct2(x, y):
    return [0, -300]
boundary.Value_def(0, 112, g, funct1)
boundary.Value_def(67, 67, g, funct2)
#boundary.Value_def(85, 85, g, funct2)
boundary.save('case/Boun_Cond.npz')
#==================================================================================================








# TEST

import matplotlib.pyplot as plt

def visualize_bc(graph, bc):
    fig, ax = plt.subplots(figsize=(8, 8))

    # --- draw mesh boundary ---
    for e in graph.edges:
        if e.is_boundary:
            v1, v2 = e.vertex
            ax.plot([v1.x, v2.x], [v1.y, v2.y],
                    color='lightgray', lw=0.8)

    # --- draw boundary conditions ---
    for v in graph.vertices:
        x, y = v.x, v.y
        kx, ky = data['Kind'][v.idx]
        bx, by = data['Boundary'][v.idx]
        #print(v.idx,'kind ',kx,ky,'BC ',bx,by)

        # displacement constraint
        if kx == 1 and ky == 1:
            ax.scatter(x, y, color='red', s=30, zorder=5)

        # force boundary
        if kx == 2 or ky == 2:
            if abs(bx) > 1e-8 or abs(by) > 1e-8:
                ax.quiver(
                    x, y,
                    bx, by,
                    angles='xy', scale_units='xy', scale=20,
                    color='blue', width=0.004, zorder=5
                )
            else:
                ax.scatter(x, y, color='green', s=10, zorder=4)

    ax.set_aspect('equal')
    ax.set_title("Boundary Condition Check")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.show()
data=np.load('case/Boun_Cond.npz')
print(data)
visualize_bc(g, data)