"""
analysis.py — 梁理论对标验证模块

用于将有限元计算结果与欧拉-伯努利梁理论的解析解进行对比。

对悬臂梁受均布载荷的情况：
    弯矩 M(x) = q·(L-x)²/2（或类似的分布形式）
    应力 σ = M·y / I_z

本脚本绘制以下对比曲线：
    biggrid_1 / smallgrid_1：不同网格密度下有限元计算的 σ_x 沿某条线的分布
    MY_IZ1：对应的梁理论应力 M·y/I_z

该模块不是主求解管线的一部分，仅用于验证有限元结果的正确性。
"""

import numpy as np

# 加载不同网格密度下的有限元应力结果
biggrid_1_data = np.load('case/biggrid1.npz')
biggrid_1 = biggrid_1_data['arr']
biggrid_2_data = np.load('case/biggrid2.npz')
biggrid_2 = biggrid_2_data['arr']
smallgrid_1_data = np.load('case/smallgrid1.npz')
smallgrid_1 = smallgrid_1_data['arr']
smallgrid_2_data = np.load('case/smallgrid2.npz')
smallgrid_2 = smallgrid_2_data['arr']

# 采样点：从 x = -1.5 到 x = 1.47，共 100 个点
X_array = [0.03 * i - 1.5 for i in range(100)]

import matplotlib.pyplot as plt

# ========== 对比组 1：弯矩 M = 30 kN·m ==========
# 梁理论公式：σ = M·y / I_z
# 截面：b = 1 m, h = 3 m → I_z = bh³/12 = 1×3³/12
MY_IZ1 = np.array([30 * 30 * (0.03 * i - 1.5) / (1 * (3 ** 3) / 12) for i in range(100)])

plt.plot(X_array, MY_IZ1, label='Beam theory (M=30)')
plt.plot(X_array, biggrid_1, label='FEM (coarse mesh)')
plt.plot(X_array, smallgrid_1, label='FEM (fine mesh)')
plt.legend()
plt.show()

# ========== 对比组 2：弯矩 M = 20 kN·m ==========
MY_IZ2 = np.array([20 * 30 * (0.03 * i - 1.5) / (1 * (3 ** 3) / 12) for i in range(100)])

plt.plot(X_array, MY_IZ2, label='Beam theory (M=20)')
plt.plot(X_array, biggrid_2, label='FEM (coarse mesh)')
plt.plot(X_array, smallgrid_2, label='FEM (fine mesh)')
plt.legend()
plt.show()
