import math
import numpy as np
from collections import defaultdict

class Vertex:
    def __init__(self, idx, x, y):
        self.idx = idx
        self.x = x
        self.y = y
        self.neighbour = set()
        self.is_boundary = False
    def define_info(self):
        self.matrix_info = {}

class Edge:
    def __init__(self, v1, v2):
        self.vertex = (v1, v2)
        self.length = 0.0
        self.angle = 0.0
        self.is_boundary = False
        self._calc()

    def _calc(self):
        v1, v2 = self.vertex
        dx = v1.x - v2.x
        dy = v1.y - v2.y
        self.length = math.hypot(dx, dy)

class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.edges = []

    def build(self, triangles):
        self.triangles = []
        edge_cnt = defaultdict(int)
        edge_obj = {}
        edge_idx = {}

        for i1, i2, i3 in triangles:
            v1 = self.vertices[i1]
            v2 = self.vertices[i2]
            v3 = self.vertices[i3]
            area = (v2.x - v1.x)*(v3.y - v1.y) - (v3.x - v1.x)*(v2.y - v1.y)
            if area < 0:
                i1, i2, i3 = i1, i3, i2
                v1, v2, v3 = v1, v3, v2

            pairs = [
                (i1, i2, v1, v2),
                (i2, i3, v2, v3),
                (i3, i1, v3, v1)
            ]

            eids = []
            for a, b, va, vb in pairs:
                key = (min(a, b), max(a, b))
                edge_cnt[key] += 1

                if key not in edge_obj:
                    e = Edge(va, vb)
                    edge_obj[key] = e
                    edge_idx[key] = len(self.edges)
                    self.edges.append(e)

                eids.append(edge_idx[key])
                va.neighbour.add(vb)
                vb.neighbour.add(va)

            self.triangles.append((i1, i2, i3, *eids))

        for (a, b), e in edge_obj.items():
            if edge_cnt[(a, b)] == 1:
                e.is_boundary = True
                e.vertex[0].is_boundary = True
                e.vertex[1].is_boundary = True

        for v in self.vertices:
            related_ids = {v.idx} | {nb.idx for nb in v.neighbour}
            v.define_info()
            v.matrix_info = {nid: np.zeros((2, 2)) for nid in related_ids}