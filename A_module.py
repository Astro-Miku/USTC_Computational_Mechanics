"""
A_module.py — 核心数据结构模块

定义了有限元网格的基本数据结构：
    Vertex  : 网格节点（顶点），记录坐标、邻接关系和矩阵分块信息
    Edge    : 网格边，记录端点、长度、是否为边界边
    Graph   : 网格图，管理所有顶点和边，提供从三角形列表构建拓扑的方法

这些类构成了后续单元刚度计算、总体组装、边界条件施加和后处理的基础。
"""

import math
import numpy as np
from collections import defaultdict

class Vertex:
    """网格节点（顶点）"""
    def __init__(self, idx, x, y):
        """
        参数
        ----
        idx : int
            节点编号（0-based）
        x, y : float
            节点坐标
        """
        self.idx = idx
        self.x = x
        self.y = y
        self.neighbour = set()     # 邻接节点集合
        self.is_boundary = False   # 是否为边界节点
    def define_info(self):
        """初始化矩阵分块存储字典。每个节点存储以自身和邻域为键的 2×2 刚度子矩阵。"""
        self.matrix_info = {}

class Edge:
    """网格边，连接两个顶点"""
    def __init__(self, v1, v2):
        """
        参数
        ----
        v1, v2 : Vertex
            边的两个端点
        """
        self.vertex = (v1, v2)
        self.length = 0.0
        self.angle = 0.0
        self.is_boundary = False
        self._calc()

    def _calc(self):
        """计算边的长度（欧氏距离）"""
        v1, v2 = self.vertex
        dx = v1.x - v2.x
        dy = v1.y - v2.y
        self.length = math.hypot(dx, dy)

class Graph:
    """网格图：管理所有顶点与边，维护三角形单元列表"""
    def __init__(self, vertices):
        """
        参数
        ----
        vertices : list of Vertex
            网格顶点列表
        """
        self.vertices = vertices
        self.edges = []

    def build(self, triangles):
        """
        从三角形列表构建网格拓扑。

        完成的步骤：
        1. 检查每个三角形的面积符号，若为负则翻转节点顺序确保逆时针取向
        2. 创建或复用三条边，建立顶点之间的邻接关系
        3. 统计每条边出现的次数：出现 1 次即为边界边

        参数
        ----
        triangles : list of tuple
            每个元素为 (i1, i2, i3)，对应 self.vertices 中的三个节点索引
        """
        self.triangles = []
        edge_cnt = defaultdict(int)    # 边出现次数计数器
        edge_obj = {}                   # (min_idx, max_idx) → Edge 对象
        edge_idx = {}                   # (min_idx, max_idx) → 边在 self.edges 中的索引

        for i1, i2, i3 in triangles:
            v1 = self.vertices[i1]
            v2 = self.vertices[i2]
            v3 = self.vertices[i3]
            # 检查三角形面积符号，面积 < 0 则翻转节点顺序（调整为逆时针）
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

            # triangle 存储: (i1, i2, i3, e12_idx, e23_idx, e31_idx)
            self.triangles.append((i1, i2, i3, *eids))

        # 标记边界边：出现次数为 1 的边为边界边
        for (a, b), e in edge_obj.items():
            if edge_cnt[(a, b)] == 1:
                e.is_boundary = True
                e.vertex[0].is_boundary = True
                e.vertex[1].is_boundary = True

        # 为每个节点初始化刚度矩阵分块存储
        for v in self.vertices:
            related_ids = {v.idx} | {nb.idx for nb in v.neighbour}
            v.define_info()
            v.matrix_info = {nid: np.zeros((2, 2)) for nid in related_ids}
