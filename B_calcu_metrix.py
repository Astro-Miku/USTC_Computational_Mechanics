"""
B_calcu_metrix.py — 单元刚度矩阵计算模块

实现了常应变三角形（CST）单元的有限元公式：
    - 形函数（shape function）对坐标的偏导数 N_ijk
    - 应变-位移矩阵 B（通过 LN 构造）
    - 弹性矩阵 D（平面应力/平面应变，通过 getD）
    - 单元刚度矩阵计算（通过边力 edge_force 和节点力 vertex_force）

理论依据：
    单元刚度矩阵 k_e = ∫_Ω Bᵀ D B dΩ
    对于 CST 单元，B 和 D 在单元内为常数，因此 k_e = Bᵀ D B · t · A
    其中 A 为三角形面积，t 为单位厚度（平面问题）

本模块采用一种等价的分解路径：
    先计算每条边上的"等效边力"，再将边力分配到节点上，
    从而得到 6×6 的单元刚度矩阵。
"""

import numpy as np
import math

def N_ijk(v1, v2, v3):
    """
    计算 CST 三角形单元形函数对坐标的偏导数。

    对于形函数 N_i(x,y), N_j(x,y), N_k(x,y)，返回：
        (∂N_i/∂x, ∂N_i/∂y, ∂N_j/∂x, ∂N_j/∂y, ∂N_k/∂x, ∂N_k/∂y)

    参数
    ----
    v1, v2, v3 : Vertex
        三角形三个顶点（逆时针顺序）

    返回
    ----
    N : tuple of 6 floats
        (Nix, Niy, Njx, Njy, Nkx, Nky)
    """
    xi = v1.x
    yi = v1.y
    xj = v2.x
    yj = v2.y
    xk = v3.x
    yk = v3.y
    # 三角形面积的两倍（2A）
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
    """
    由形函数偏导数构造应变-位移矩阵 B（3×6）。

    B 矩阵将节点位移 {u_i, v_i, u_j, v_j, u_k, v_k} 映射为应变 {ε_x, ε_y, γ_xy}:
        ε = B · d

    参数
    ----
    N : tuple of 6 floats
        由 N_ijk 返回的形函数偏导数

    返回
    ----
    B : ndarray, shape (3, 6)
    """
    Nix, Niy, Njx, Njy, Nkx, Nky = N
    B = np.array([
        [Nix, 0,   Njx, 0,   Nkx, 0],
        [0,   Niy, 0,   Njy, 0,   Nky],
        [Niy, Nix, Njy, Njx, Nky, Nkx]
    ])
    return B

def getD(E, niu, is_stress):
    """
    构造平面问题的弹性矩阵 D（3×3）。

    应力-应变关系：σ = D · ε

    参数
    ----
    E : float
        杨氏模量
    niu : float
        泊松比
    is_stress : int
        0 = 平面应变，1 = 平面应力

    返回
    ----
    D : ndarray, shape (3, 3)
    """
    if is_stress == 0:  # 0 = 平面应变, 1 = 平面应力
        # 平面应变：等效杨氏模量和泊松比的转换
        E, niu = E / (1 - niu * niu), niu / (1 - niu)
    # 通用形式（平面应力或转换后的平面应变）
    D = E / (1 - niu ** 2) * np.array([
        [1, niu, 0],
        [niu, 1, 0],
        [0, 0, (1 - niu) / 2]
    ])
    return D

def normal(v1, v2):
    """
    计算从 v1 指向 v2 的边的单位外法向量。

    对于以逆时针顺序遍历的三角形，该法线指向单元外侧。

    参数
    ----
    v1, v2 : Vertex
        边的两个端点

    返回
    ----
    n12 : tuple (nx, ny)
        单位外法向量分量
    """
    dx12 = v2.x - v1.x
    dy12 = v2.y - v1.y
    L12 = math.hypot(dx12, dy12)
    n12 = (dy12 / L12, -dx12 / L12)
    return n12

def edge_force(graph, tri_idx, E, niu, is_stress):
    """
    计算三角形单元各边上的等效边力（6×3 矩阵）。

    对三角形的三条边分别计算法向应力分量在边上的积分，
    得到每条边上 x 和 y 方向的力。结果是一个 6×3 的矩阵，
    6 行对应三条边各自的 x/y 分量，3 列对应 σ_x, σ_y, τ_xy。

    参数
    ----
    graph : Graph
        网格图
    tri_idx : int
        三角形在 graph.triangles 中的索引
    E : float
        杨氏模量
    niu : float
        泊松比
    is_stress : int
        0 = 平面应变, 1 = 平面应力

    返回
    ----
    ndarray, shape (6, 3)
        边力矩阵
    """
    tri = graph.triangles[tri_idx]
    v1_idx, v2_idx, v3_idx, e12_idx, e23_idx, e31_idx = tri
    v1, v2, v3 = graph.vertices[v1_idx], graph.vertices[v2_idx], graph.vertices[v3_idx]
    e12, e23, e31 = graph.edges[e12_idx], graph.edges[e23_idx], graph.edges[e31_idx]

    # D·B: 应力-位移关系矩阵 (3×6)
    DB = np.dot(getD(E, niu, is_stress), LN(N_ijk(v1, v2, v3)))

    # 三条边的单位外法向量
    outnormal = [normal(v1, v2), normal(v2, v3), normal(v3, v1)]

    # 边力转换矩阵 T_edge (6×3)
    # 将应力分量 (σ_x, σ_y, τ_xy) 转换为各边的 x,y 方向力
    # 每行：对应一条边的一个方向分量 × 边长度
    edge_force_trans = np.array([
        [outnormal[0][0] * e12.length, 0,                          outnormal[0][1] * e12.length],  # 边12-x
        [0,                           outnormal[0][1] * e12.length, outnormal[0][0] * e12.length],  # 边12-y
        [outnormal[1][0] * e23.length, 0,                          outnormal[1][1] * e23.length],  # 边23-x
        [0,                           outnormal[1][1] * e23.length, outnormal[1][0] * e23.length],  # 边23-y
        [outnormal[2][0] * e31.length, 0,                          outnormal[2][1] * e31.length],  # 边31-x
        [0,                           outnormal[2][1] * e31.length, outnormal[2][0] * e31.length],  # 边31-y
    ])
    return np.dot(edge_force_trans, DB)

def vertex_force(graph, tri_idx, E, niu, is_stress):
    """
    计算三角形单元刚度矩阵 k_e（6×6）。

    将边力分配到三个节点上，得到单元刚度矩阵。
    对于每条边，其上的边力平均分配给该边的两个端点。
    因此节点力的转换关系为：
        k_e = T_vertex · T_edge · D · B

    其中 T_vertex 将 6 个边力分量映射为 6 个节点力分量（每个节点 x, y）
    分配规则：每条边上的力平分给两个端点。

    参数
    ----
    graph : Graph
        网格图
    tri_idx : int
        三角形在 graph.triangles 中的索引
    E : float
        杨氏模量
    niu : float
        泊松比
    is_stress : int
        0 = 平面应变, 1 = 平面应力

    返回
    ----
    k_e : ndarray, shape (6, 6)
        单元刚度矩阵，行/列顺序为:
        [v1_x, v1_y, v2_x, v2_y, v3_x, v3_y]
    """
    edge_matrix = edge_force(graph, tri_idx, E, niu, is_stress)  # (6×3)

    # 节点力转换矩阵 T_vertex (6×6)
    # 每行对应一个节点的一个方向分量
    # 例如：节点1的x方向力 = 0.5*(边12的x力) + 0.5*(边31的x力)
    vertex_trans = np.array([
        [0.5, 0,   0,   0,   0.5, 0  ],   # 节点1-x: 边12和边31各贡献一半
        [0,   0.5, 0,   0,   0,   0.5],   # 节点1-y
        [0.5, 0,   0.5, 0,   0,   0  ],   # 节点2-x: 边12和边23各贡献一半
        [0,   0.5, 0,   0.5, 0,   0  ],   # 节点2-y
        [0,   0,   0.5, 0,   0.5, 0  ],   # 节点3-x: 边23和边31各贡献一半
        [0,   0,   0,   0.5, 0,   0.5],   # 节点3-y
    ])
    return np.dot(vertex_trans, edge_matrix)
