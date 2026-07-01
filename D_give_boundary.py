"""
D_give_boundary.py — 边界条件施加与求解准备模块

提供 apply_displacement_constraints 函数，将位移边界条件施加到
总体刚度矩阵 K 和载荷向量 F 上。

边界条件施加方法：惩罚法（penalty method）
    对于 Dirichlet 边界 u_i = ū_i：
        - K 的第 i 行全部置零
        - K[i][i] 设为 1
        - F[i] 设为 ū_i
    对于 Neumann 边界（力边界），直接将边界值写入 F。
"""

import numpy as np
from C_build_whole import calcu_cell, build_whole
from scipy.sparse import coo_matrix

def apply_displacement_constraints(g, E, nu, BCfile, is_force=1):
    """
    组装全局刚度矩阵并施加边界条件。

    工作流程：
    1. 计算所有单元的刚度矩阵（calcu_cell）
    2. 组装为全局稀疏矩阵 K（build_whole + COO → CSR）
    3. 从 BCfile（Boun_Cond.npz）读取 Kind 和 Boundary 数组
    4. 对每个自由度：
       - Kind[i//2][i%2] == 1：位移边界，使用惩罚法施加
       - Kind[i//2][i%2] == 2：力边界，将载荷值填入 F

    参数
    ----
    g : Graph
        网格图（已完成 build）
    E : float
        杨氏模量
    nu : float
        泊松比
    BCfile : str
        边界条件文件路径（Boun_Cond.npz）
    is_force : int
        0 = 平面应变, 1 = 平面应力（默认平面应力）

    返回
    ----
    K : csr_matrix
        施加边界条件后的全局刚度矩阵（CSR 格式）
    F : ndarray, shape (n_dof,)
        载荷向量
    """
    data_BC = np.load(BCfile)
    calcu_cell(g, E, nu, is_force)
    rows, cols, data, shape = build_whole(g)
    K = coo_matrix((data, (rows, cols)), shape=shape).tocsr()
    n_dof = shape[0]
    F = np.zeros(n_dof)

    for i in range(n_dof):
        F[i] = data_BC['Boundary'][i // 2][i % 2]
        if data_BC['Kind'][i // 2][i % 2] == 1:
            # 惩罚法施加位移边界条件
            K[i, :] = 0       # 整行置零
            K[i, i] = 1       # 对角元设为 1（等价于 u_i = F[i]）
    return K, F
