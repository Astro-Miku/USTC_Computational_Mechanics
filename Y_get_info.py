"""
Y_get_info.py — 网格数据加载模块

从 npz 格式的网格文件加载节点坐标和三角形单元索引，
构造 Vertex 对象列表和三角形索引列表。

npz 文件格式（由 Z_readmphtxt.py 或 Z_readtxt.py 生成）：
    coords  : (N, 2) 浮点数组 — 节点坐标
    tris    : (M, 3) 整数数组 — 三角形顶点索引（0-based）
    n_nodes : int — 节点数
    n_tris  : int — 三角形单元数
"""

import numpy as np
from A_module import Vertex

def load_grid_from_npz(npz_file):
    """
    从 npz 文件加载网格数据。

    参数
    ----
    npz_file : str
        npz 文件路径

    返回
    ----
    verts : list of Vertex
        网格顶点列表（0-based 编号）
    tris_list : list of tuple
        三角形列表，每个元素为 (i1, i2, i3)
    """
    data = np.load(npz_file)
    coords = data['coords']  # N x 2
    tris = data['tris']      # M x 3

    # 构造 Vertex 对象列表
    verts = []
    for i in range(len(coords)):
        x, y = coords[i]
        verts.append(Vertex(i, float(x), float(y)))

    # 转换为元组列表
    tris_list = []
    for i in range(len(tris)):
        a, b, c = tris[i]
        tris_list.append((int(a), int(b), int(c)))

    return verts, tris_list

if __name__ == "__main__":
    npz_file = "grid.npz"
    verts, tris = load_grid_from_npz(npz_file)
