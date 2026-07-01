"""
E_main.py — 有限元求解主程序

完整的求解流程：
    1. 加载网格（grid.npz）并构建拓扑图
    2. 组装全局刚度矩阵并施加边界条件（调用 apply_displacement_constraints）
    3. 稀疏求解 K·U = F，得到位移场 U
    4. 可视化：位移 u 和 v 云图
    5. 保存位移结果到 DisplacementResult.npz

运行前需要先生成边界条件文件：
    cd case && python ../W_give_BC.py
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from A_module import Vertex, Graph
from D_give_boundary import apply_displacement_constraints
from Y_get_info import load_grid_from_npz

def solve_five_point():
    """
    执行完整的有限元求解流程。

    返回
    ----
    U : ndarray, shape (2N,)
        位移向量，U[2i] 和 U[2i+1] 分别为节点 i 的 x 和 y 方向位移
    """
    # 1. 加载网格并构建拓扑
    npz_file = "case/grid.npz"
    verts, tris = load_grid_from_npz(npz_file)
    g = Graph(verts)
    g.build(tris)

    # 2. 单元刚度计算并组装，施加边界条件
    E, nu = 200e9, 0.3
    BCfile = "case/Boun_Cond.npz"

    K, F = apply_displacement_constraints(
        g, E, nu, BCfile, is_force=1
    )
    # 稀疏求解 K·U = F
    U = spsolve(K, F)

    # 3. 提取可视化数据
    bc_data = np.load(BCfile, allow_pickle=True)
    kind = bc_data['Kind']
    boundary = bc_data['Boundary']

    # 提取所有节点的坐标和位移分量
    x_coords = [v.x for v in g.vertices]
    y_coords = [v.y for v in g.vertices]
    u_disp = [U[2*i] for i in range(len(g.vertices))]
    v_disp = [U[2*i+1] for i in range(len(g.vertices))]

    # 4. 创建位移场可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 构造 matplotlib 三角剖分
    triangles = np.array([[t[0], t[1], t[2]] for t in g.triangles])
    triang = tri.Triangulation(x_coords, y_coords, triangles)

    # 图1: u 方向位移云图
    ax2 = axes[0]
    tpc_u = ax2.tripcolor(triang, u_disp, cmap='viridis', shading='gouraud')
    plt.colorbar(tpc_u, ax=ax2, label='u displacement')
    ax2.set_aspect('equal')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Displacement u (x-direction)')

    # 图2: v 方向位移云图
    ax3 = axes[1]
    tpc_v = ax3.tripcolor(triang, v_disp, cmap='viridis', shading='gouraud')
    plt.colorbar(tpc_v, ax=ax3, label='v displacement')
    ax3.set_aspect('equal')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Displacement v (y-direction)')

    plt.tight_layout()
    plt.savefig("case/result.png", dpi=300, bbox_inches='tight')
    plt.close()

    return U


if __name__ == "__main__":
    U = solve_five_point()
    # 保存位移结果供后处理使用
    np.savez('case/DisplacementResult.npz', U=U)
