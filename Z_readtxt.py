import numpy as np

def read_grid_txt(file_path):
    """
    读取grid.txt文件，提取节点坐标和三角形单元
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 找到坐标和三角形数据的起始位置
    coord_start = None
    tri_start = None
    
    for i, line in enumerate(lines):
        if '% Coordinates' in line:
            coord_start = i + 1
        elif '% Elements (triangles)' in line:
            tri_start = i + 1
            break
    
    # 提取坐标数据
    coords = []
    for i in range(coord_start, tri_start - 1):
        line = lines[i].strip()
        if line:  # 跳过空行
            parts = line.split()
            if len(parts) >= 2:
                x, y = map(float, parts[:2])
                coords.append([x, y])
    
    # 提取三角形数据
    tris = []
    for i in range(tri_start, len(lines)):
        line = lines[i].strip()
        if line:  # 跳过空行
            parts = line.split()
            if len(parts) >= 3:
                # 注意：COMSOL索引从1开始，转换为从0开始
                a, b, c = map(int, parts[:3])
                tris.append([a - 1, b - 1, c - 1])  # 转换为0-based索引
    
    return np.array(coords), np.array(tris)

def save_grid_to_npz(txt_file, npz_file):
    """
    读取txt文件并保存为npz文件
    """
    coords, tris = read_grid_txt(txt_file)
    
    # 保存到npz文件
    np.savez(npz_file, 
             coords=coords, 
             tris=tris,
             n_nodes=len(coords),
             n_tris=len(tris))
    
    print(f"已保存到 {npz_file}")
    print(f"节点数: {len(coords)}")
    print(f"三角形数: {len(tris)}")
    
    return coords, tris

# 使用示例
if __name__ == "__main__":
    txt_file = "case/grid.txt"  # 你的txt文件路径
    npz_file = "case/grid.npz"  # 输出的npz文件路径
    
    coords, tris = save_grid_to_npz(txt_file, npz_file)