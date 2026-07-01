"""
G_get_displacement.py — 位移加载与插值模块

提供两个核心功能：
    load_result               : 加载网格和位移结果
    interpolate_displacement  : 在任意坐标 (x, y) 处通过形函数插值位移

插值方法：利用 CST 单元的线性形函数，在包含目标点的三角形内
通过节点位移进行重心坐标插值。
"""

import numpy as np
from A_module import Vertex, Edge, Graph
from F_find_tri import find_triangle_of_point
from Y_get_info import load_grid_from_npz

def load_result(graph_path, U_path):
    """
    加载网格文件和位移结果。

    参数
    ----
    graph_path : str
        网格 npz 文件路径（如 'case/grid.npz'）
    U_path : str
        位移结果 npz 文件路径（如 'case/DisplacementResult.npz'）

    返回
    ----
    g : Graph
        包含拓扑信息的网格图
    U : ndarray, shape (2N,)
        位移向量
    """
    verts, tris = load_grid_from_npz(graph_path)
    data = np.load(U_path)
    U = data['U']
    g = Graph(verts)
    g.build(tris)
    return g, U

def interpolate_displacement(graph, U, px, py):
    """
    在任意坐标 (px, py) 处插值位移。

    采用 CST 单元的线性形函数插值：找到包含该点的三角形，
    计算重心坐标 w1, w2, w3，然后加权求和节点位移。

    参数
    ----
    graph : Graph
        网格图
    U : ndarray, shape (2N,)
        位移向量
    px, py : float
        目标点坐标

    返回
    ----
    (u, v) : tuple of float 或 None
        插值得到的 x 和 y 方向位移分量
        若点在网格外则返回 None
    """
    # 找到包含该点的三角形
    tri = find_triangle_of_point(graph, px, py)
    if tri is None:
        return None

    i1, i2, i3, *_ = tri
    v1, v2, v3 = graph.vertices[i1], graph.vertices[i2], graph.vertices[i3]

    x1, y1 = v1.x, v1.y
    x2, y2 = v2.x, v2.y
    x3, y3 = v3.x, v3.y

    # 计算重心坐标 w1, w2, w3（基于三角形面积比）
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    w1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
    w2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
    w3 = 1 - w1 - w2

    # 加权插值节点位移
    u = w1 * U[2 * i1] + w2 * U[2 * i2] + w3 * U[2 * i3]
    v = w1 * U[2 * i1 + 1] + w2 * U[2 * i2 + 1] + w3 * U[2 * i3 + 1]

    return u, v
