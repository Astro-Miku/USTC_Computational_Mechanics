"""
X_GeometryVisual.py — 边界拓扑提取与可视化模块

从网格图中提取所有边界边，按连通性分组为闭合/半闭合的边界环（loop），
并将每个环存储为链表结构（link list），供 W_give_BC.py 沿边界传播条件。

输出文件：
    case/boundary.png  — 边界环可视化图
    case/boun_link.npz — 包含 head（各环起点）和 link（节点→下一节点映射）

工作流程：
    1. 从原 Graph 中筛选出 is_boundary=True 的边
    2. 构造仅含边界边的子图 boun_g
    3. extract_all_loops：遍历邻接表，找到所有边界环
    4. visualize_and_save：可视化并保存为 npz
"""

from A_module import Vertex, Edge, Graph
from Y_get_info import load_grid_from_npz
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ========== 1. 构建仅含边界边的子图 ==========

npz_file = "case/grid.npz"
verts, tris = load_grid_from_npz(npz_file)

g = Graph(verts)
g.build(tris)

new_vert, _ = load_grid_from_npz(npz_file)

# 创建仅含边界边的图 boun_g
boun_g = Graph(new_vert)

idx_map = {v.idx: v for v in new_vert}

boun_edges = []
for edge in g.edges:
    if edge.is_boundary:
        va_idx = edge.vertex[0].idx
        vb_idx = edge.vertex[1].idx

        new_va = idx_map[va_idx]
        new_vb = idx_map[vb_idx]

        new_edge = Edge(new_va, new_vb)
        new_edge.length = edge.length
        new_edge.angle = edge.angle
        new_edge.is_boundary = True

        new_va.neighbour.add(new_vb)
        new_vb.neighbour.add(new_va)

        boun_edges.append(new_edge)

boun_g.edges = boun_edges

# ========== 2. 边界环提取 ==========

def extract_all_loops(graph):
    """
    从边界图中提取所有边界环（loop）。

    遍历每个边界节点的邻接表，沿相邻节点追踪直到回到起点，
    识别所有闭合的边界环。每个环是一个节点索引列表。

    参数
    ----
    graph : Graph
        仅包含边界边的图（boun_g）

    返回
    ----
    loops : list of list
        每个元素为一个边界环的节点索引列表（按遍历顺序）
    """
    neighbors = defaultdict(list)
    for edge in graph.edges:
        if edge.is_boundary:
            va, vb = edge.vertex
            neighbors[va.idx].append(vb.idx)
            neighbors[vb.idx].append(va.idx)

    visited = set()
    loops = []

    for idx in list(neighbors.keys()):
        if idx in visited:
            continue

        loop = []
        current = idx
        prev = None

        while True:
            loop.append(current)
            visited.add(current)

            # 排除来路，选择下一节点
            next_candidates = [n for n in neighbors[current] if n != prev]
            if not next_candidates:
                break

            next_idx = next_candidates[0]

            # 回到起点且环长度大于 2，则为闭合环
            if next_idx == idx and len(loop) > 2:
                loop.append(next_idx)
                visited.add(next_idx)
                break

            prev = current
            current = next_idx

        if len(loop) > 2:
            loops.append(loop)

    return loops

# ========== 3. 可视化并保存 ==========

def visualize_and_save(graph, loops):
    """
    可视化所有边界环并保存为 npz 文件。

    对每个环绘制折线、标注节点索引、标注遍历方向。

    boun_link.npz 格式：
        nlink : int — 环的数量
        head   : ndarray — 各环起始节点索引
        link   : ndarray — (u, v) 边对，表示环中相邻节点的链表关系

    参数
    ----
    graph : Graph
        边界子图
    loops : list of list
        extract_all_loops 的返回值
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(loops), 1)))

    # 绘制每个环
    for i, loop_order in enumerate(loops):
        xs = [graph.vertices[idx].x for idx in loop_order]
        ys = [graph.vertices[idx].y for idx in loop_order]
        xs.append(xs[0])  # 闭合环
        ys.append(ys[0])

        # 绘制环的折线
        ax.plot(xs, ys, '-', linewidth=2, marker='o', markersize=6,
               markerfacecolor=colors[i], label=f'Loop {i}', color=colors[i])

        # 标注节点索引
        for idx in loop_order:
            v = graph.vertices[idx]
            ax.annotate(f'{idx}', (v.x, v.y),
                       xytext=(3, 3), textcoords='offset points',
                       fontsize=3, fontweight='bold')

        # 标注遍历方向（箭头）
        for j in range(len(loop_order) - 1):
            x1, y1 = graph.vertices[loop_order[j]].x, graph.vertices[loop_order[j]].y
            x2, y2 = graph.vertices[loop_order[j+1]].x, graph.vertices[loop_order[j+1]].y
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5, alpha=0.7))

    # 设置图的外观
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Boundary Loops Visualization')
    ax.grid(True, alpha=0.3)

    if loops:
        ax.legend()

    # 保存图片
    plt.savefig('case/boundary.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 准备 npz 数据
    nlink = len(loops)
    head = [loop[0] for loop in loops]  # 各环起始节点
    link_data = []

    for loop in loops:
        n = len(loop)
        # 仅保存到最后一个元素，不链回起点
        for i in range(n - 1):
            current_idx = loop[i]
            next_idx = loop[i + 1]
            link_data.append([current_idx, next_idx])

    # 转换为 numpy 数组
    head_array = np.array(head)
    link_array = np.array(link_data)

    # 保存到 npz 文件
    np.savez('case/boun_link.npz',
             nlink=nlink,
             head=head_array,
             link=link_array)

    return head_array, link_array

# ========== 执行 ==========
loops = extract_all_loops(boun_g)
head_array, link_array = visualize_and_save(boun_g, loops)
