"""
C_build_whole.py — 总体刚度矩阵组装模块

负责：
    calcu_cell : 遍历所有三角形单元，计算每个单元的刚度矩阵，
                 并累加到各节点的 matrix_info 分块中
    build_whole: 将所有节点的 2×2 分块矩阵组装为全局稀疏矩阵（COO 格式），
                 返回 CSR 组装所需的数据

DOF 映射：节点 i 对应全局自由度 (2i, 2i+1)，分别表示 x 和 y 方向位移。
"""

import numpy as np
import math
from B_calcu_metrix import vertex_force

def calcu_cell(graph, E, niu, is_force):
    """
    遍历所有三角形单元，计算单元刚度矩阵并累加到节点分块存储中。

    每个单元给出 6×6 的刚度矩阵 k_e，按节点分块（每个 2×2）：
        [k_ii  k_ij  k_ik]
        [k_ji  k_jj  k_jk]
        [k_ki  k_kj  k_kk]

    分别累加到对应节点的 matrix_info 字典中。

    参数
    ----
    graph : Graph
        网格图（已完成 build）
    E : float
        杨氏模量
    niu : float
        泊松比
    is_force : int
        0 = 平面应变, 1 = 平面应力

    返回
    ----
    None（结果直接写入 graph.vertices[i].matrix_info）
    """
    lentri = len(graph.triangles)
    for i in range(lentri):
        single_matrix = vertex_force(graph, i, E, niu, is_force)
        """
            Assemble 6x6 element stiffness matrix into global matrix.
            Each 2x2 block maps to DOF pairs (2*i, 2*j) for vertices i1,i2,i3.
        """
        i1, i2, i3, _, _, _ = graph.triangles[i]
        # k_ii, k_ij, k_ik —— 节点 i1 的贡献
        graph.vertices[i1].matrix_info[i1] += single_matrix[0:2, 0:2]
        graph.vertices[i1].matrix_info[i2] += single_matrix[0:2, 2:4]
        graph.vertices[i1].matrix_info[i3] += single_matrix[0:2, 4:6]

        # k_ji, k_jj, k_jk —— 节点 i2 的贡献
        graph.vertices[i2].matrix_info[i1] += single_matrix[2:4, 0:2]
        graph.vertices[i2].matrix_info[i2] += single_matrix[2:4, 2:4]
        graph.vertices[i2].matrix_info[i3] += single_matrix[2:4, 4:6]

        # k_ki, k_kj, k_kk —— 节点 i3 的贡献
        graph.vertices[i3].matrix_info[i1] += single_matrix[4:6, 0:2]
        graph.vertices[i3].matrix_info[i2] += single_matrix[4:6, 2:4]
        graph.vertices[i3].matrix_info[i3] += single_matrix[4:6, 4:6]
    return

def build_whole(graph):
    """
    将存储在节点分块 matrix_info 中的子矩阵组装为全局稀疏格式。

    每个节点 i 存储了以 j 为键的 2×2 矩阵 K_{ij}（DOF 2i,2i+1 × 2j,2j+1）。
    本函数将这些分块展开为 COO 三元组 (row, col, data)。

    参数
    ----
    graph : Graph
        已完成 calcu_cell 的网格图

    返回
    ----
    rows : ndarray (nnz,)
        非零元行索引
    cols : ndarray (nnz,)
        非零元列索引
    data : ndarray (nnz,)
        非零元数值
    shape : tuple (n_dof, n_dof)
        全局矩阵维度
    """
    n_v = len(graph.vertices)
    n_dof = 2 * n_v      # 每个节点 2 个自由度 (x, y)

    rows_list = []
    cols_list = []
    data_list = []
    # 2×2 分块内的局部行列索引
    local_rows = np.array([0, 0, 1, 1])
    local_cols = np.array([0, 1, 0, 1])

    for i, v in enumerate(graph.vertices):
        for j, block in v.matrix_info.items():
            block_flat = block.ravel()       # 展开 2×2 为 4 元素
            mask = block_flat != 0

            if not np.any(mask):
                continue

            # 全局行列偏移
            base_row = 2 * i
            base_col = 2 * j

            nonzero_idx = np.where(mask)[0]
            rows = base_row + local_rows[nonzero_idx]
            cols = base_col + local_cols[nonzero_idx]
            vals = block_flat[nonzero_idx]

            rows_list.append(rows)
            cols_list.append(cols)
            data_list.append(vals)

    # 合并所有分块的数据
    if rows_list:
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        data = np.concatenate(data_list)
    else:
        rows, cols, data = np.array([]), np.array([]), np.array([])

    return rows, cols, data, (n_dof, n_dof)
