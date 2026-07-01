"""
W_give_BC.py — 边界条件定义模块

提供 BC 类，用于定义位移和力边界条件并通过边界链表沿边界传播。
支持：
    - BC_def   : 沿边界链设置 Kind 标记（1=位移, 2=力）
    - Value_def: 沿边界链计算并设置边界值（位移值或力积分值）
    - save     : 将边界条件保存为 Boun_Cond.npz

运行方式（作为脚本）：
    cd case && python ../W_give_BC.py

边界条件通过 boun_link.npz 中的链表结构沿边界传播：
    link_dict[p] = q  表示边界边从节点 p 连向节点 q。
"""

from Y_get_info import load_grid_from_npz
from A_module import Vertex, Edge, Graph
import numpy as np
import math

class BC:
    """边界条件管理器"""
    def __init__(self, verts):
        """
        参数
        ----
        verts : list of Vertex
            网格顶点列表
        """
        ids = {vert.idx for vert in verts}
        self.len = len(ids)
        self.kind = {nid: [2, 2] for nid in ids}        # 默认：力边界（1=位移, 2=力）
        self.boundary = {nid: [0.0, 0.0] for nid in ids}

    def BC_def(self, start, end, constrain_x=1, constrain_y=1):
        """
        沿边界链设置边界条件类型。

        从 start 节点出发，沿 link_dict 遍历到 end 节点，
        将途经节点的 Kind 设为指定的约束类型。

        参数
        ----
        start : int
            起始节点索引
        end : int
            终止节点索引（含 end 本身）
        constrain_x : int
            x 方向约束类型（1=位移, 2=力）
        constrain_y : int
            y 方向约束类型（1=位移, 2=力）
        """
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
        """
        沿边界链计算并设置边界值。

        对于位移边界（Kind=1）：在节点位置计算 funct 的值
        对于力边界（Kind=2）  ：在边中点计算 funct 的值，
                               沿边积分后平均分配到两端节点。

        力积分方式：funct 返回力密度 [fx, fy]（单位长度力），
                   乘以边长得到边上的总力，再平分给两个端点。

        参数
        ----
        start : int
            起始节点索引
        end : int
            终止节点索引
        graph : Graph
            网格图
        funct : callable
            函数 f(x, y) → [val_x, val_y]，
            返回力密度（力边界）或位移值（位移边界）
        """
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
            # 位移边界：直接在节点处取值
            if kx == 1 or ky == 1:
                val_x, val_y = funct(vp.x, vp.y)
                if kx == 1:  # x 方向有位移约束
                    self.boundary[p][0] = val_x
                if ky == 1:  # y 方向有位移约束
                    self.boundary[p][1] = val_y
            # 力边界：沿边积分并平均分配
            if kx == 2 or ky == 2:
                mid_x = (vp.x + vq.x) / 2.0
                mid_y = (vp.y + vq.y) / 2.0
                fx, fy = funct(mid_x, mid_y)     # 单位长度力（力密度）
                dx = vp.x - vq.x
                dy = vp.y - vq.y
                length = math.hypot(dx, dy)       # 边长度
                Fx = fx * length                   # 边上的总力
                Fy = fy * length
                if kx == 2:
                    force_accum[p][0] += Fx / 2.0  # 平分给两端点
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
        # 将累积的力载荷写入 boundary
        for nid in force_accum:
            kx, ky = self.kind[nid]
            if kx == 2:
                self.boundary[nid][0] = force_accum[nid][0]
            if ky == 2:
                self.boundary[nid][1] = force_accum[nid][1]

    def save(self, name):
        """
        将边界条件保存为 npz 文件。

        输出格式：
            Kind     : (N, 2) 整数数组，1=位移, 2=力
            Boundary : (N, 2) 浮点数组，对应的约束值或载荷值

        参数
        ----
        name : str
            输出文件路径（如 'case/Boun_Cond.npz'）
        """
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

# ========== 脚本执行入口 ==========
npz_file = "case/grid.npz"
verts, tris = load_grid_from_npz(npz_file)
g = Graph(verts)
g.build(tris)
# 加载边界链表：link_dict[p] = q 表示边界边从 p 指向 q
data = np.load('case/boun_link.npz')
link_dict = {u: v for u, v in data['link']}

boundary = BC(verts)

# USER
#==================================================================================================
# 定义位移边界：右侧边界 32→1 和底部边界 157→143 的 x,y 方向均固定
boundary.BC_def(32, 1, constrain_x=1, constrain_y=1)
boundary.BC_def(157, 143, constrain_x=1, constrain_y=1)
#boundary.BC_def(0, 0,constrain_x=1,constrain_y=1)
def funct1(x, y):
    return [0, -500]
def funct2(x, y):
    return [0, -300]
# 顶部边界 0→112：施加 y 方向分布力 -500 N/m
boundary.Value_def(0, 112, g, funct1)
# 点 67：集中力 -300 N
boundary.Value_def(67, 67, g, funct2)
#boundary.Value_def(85, 85, g, funct2)
boundary.save('case/Boun_Cond.npz')
#==================================================================================================






# TEST

import matplotlib.pyplot as plt

def visualize_bc(graph, bc):
    """可视化边界条件检验图"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # --- 绘制网格边界 ---
    for e in graph.edges:
        if e.is_boundary:
            v1, v2 = e.vertex
            ax.plot([v1.x, v2.x], [v1.y, v2.y],
                    color='lightgray', lw=0.8)

    # --- 绘制边界条件标记 ---
    for v in graph.vertices:
        x, y = v.x, v.y
        kx, ky = data['Kind'][v.idx]
        bx, by = data['Boundary'][v.idx]

        # 位移约束节点（红色）
        if kx == 1 and ky == 1:
            ax.scatter(x, y, color='red', s=30, zorder=5)

        # 力边界节点（蓝色箭头）
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

data = np.load('case/Boun_Cond.npz')
print(data)
visualize_bc(g, data)
