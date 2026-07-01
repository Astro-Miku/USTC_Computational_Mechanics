# 弹性力学二维有限元分析

## 项目简介

本项目实现了一个**线弹性二维有限元求解器**，采用常应变三角形（CST）单元，使用 COMSOL 生成的网格进行求解。完整的工作流程包括：网格导入 → 总刚组装 → 方程求解 → 后处理（位移、应力、主应力、von Mises 等）。

代码使用纯 Python 编写，依赖 `numpy`、`scipy` 和 `matplotlib`，结构清晰、模块化程度高，适合学习有限元方法的同学阅读和修改。

## 背景

本项目是中国科学技术大学 **近代力学系《计算力学基础》课程**的上机考试代码案例。由 **Wang C.Y.** 编写并授权开源。

希望这份代码能够给后来的学弟学妹们提供帮助，作为学习有限元方法和计算力学的参考。

## 快速开始

### 安装依赖

```bash
pip install numpy scipy matplotlib
```

### 运行求解

```bash
cd case
python ../W_give_BC.py     # 设置边界条件
python ../E_main.py         # 求解并保存结果
```

### 后处理

```bash
python ../test.py           # 位移、应力、主应力、von Mises 后处理
```

## 项目结构

```
├── A_module.py              # 核心数据结构：Vertex、Edge、Graph
├── B_calcu_metrix.py        # 单元刚度矩阵计算
├── C_build_whole.py         # 总刚组装（稀疏矩阵）
├── D_give_boundary.py       # 施加位移边界条件
├── E_main.py                # 求解主程序
├── F_find_tri.py            # 点在三角形网格中的定位
├── G_get_displacement.py    # 任意点位移插值
├── H_post_process.py        # 后处理（27个函数）
├── W_give_BC.py             # 边界条件定义
├── X_GeometryVisual.py      # 边界拓扑提取与可视化
├── Y_get_info.py            # 网格数据加载
├── Z_readmphtxt.py          # COMSOL .mphtxt 解析
├── Z_readtxt.py             # 通用 .txt 网格解析
├── test.py                  # 后处理示例脚本
├── analysis.py              # 梁理论验证（用于对标）
└── case/
    ├── grid.mphtxt           # COMSOL 网格（158节点，256单元）
    ├── grid.npz              # 转换后的网格数据
    ├── boun_link.npz         # 边界链表
    ├── Boun_Cond.npz         # 边界条件值
    └── DisplacementResult.npz # 求解结果
```

## 关键设计

- **边界条件**：使用惩罚法施加位移约束，类型由 `Kind` 矩阵标记（1=位移，2=力）
- **全局矩阵**：以 COO 格式组装后转换为 CSR 格式
- **应力状态**：默认平面应力（`is_stress=1`），可切换为平面应变
- **边界遍历**：边界边以链表形式存储，支持沿边界进行积分与采样


---

**中国科学技术大学 · 近代力学系 · 《计算力学基础》**
