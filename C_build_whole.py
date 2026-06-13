import numpy as np
import math
from B_calcu_metrix import vertex_force
def calcu_cell(graph,E,niu,is_force):
    lentri=len(graph.triangles)
    #whole_matrix=np.array(2*len(graph.vertices),2*len(graph.vertices))
    for i in range(lentri):
        single_matrix=vertex_force(graph,i,E,niu,is_force)
        """
            target: assemble
            
            force:  6×6
              ↓
            place:
                    2*i1        2*i1+1      2*i2        2*i2+1      2*i3        2*i3+1
            2*i1      *           *           *           *           *           *
            2*i1+1    *           *           *           *           *           *
            2*i2      *           *           *           *           *           *
            2*i2+1    *           *           *           *           *           *
            2*i3      *           *           *           *           *           *
            2*i3+1    *           *           *           *           *           *
        """
        i1, i2, i3, _, _, _=graph.triangles[i]
        graph.vertices[i1].matrix_info[i1]+=single_matrix[0:2,0:2]
        graph.vertices[i1].matrix_info[i2]+=single_matrix[0:2,2:4]
        graph.vertices[i1].matrix_info[i3]+=single_matrix[0:2,4:6]
        
        graph.vertices[i2].matrix_info[i1]+=single_matrix[2:4,0:2]
        graph.vertices[i2].matrix_info[i2]+=single_matrix[2:4,2:4]
        graph.vertices[i2].matrix_info[i3]+=single_matrix[2:4,4:6]
        
        graph.vertices[i3].matrix_info[i1]+=single_matrix[4:6,0:2]
        graph.vertices[i3].matrix_info[i2]+=single_matrix[4:6,2:4]
        graph.vertices[i3].matrix_info[i3]+=single_matrix[4:6,4:6]
    return
def build_whole(graph):
    n_v = len(graph.vertices)
    n_dof = 2 * n_v
    
    rows_list = []
    cols_list = []
    data_list = []
    local_rows = np.array([0, 0, 1, 1])
    local_cols = np.array([0, 1, 0, 1])
    for i, v in enumerate(graph.vertices):
        for j, block in v.matrix_info.items():
            block_flat = block.ravel()
            mask = block_flat != 0
            
            if not np.any(mask):
                continue
                
            base_row = 2 * i
            base_col = 2 * j
            
            nonzero_idx = np.where(mask)[0]
            rows = base_row + local_rows[nonzero_idx]
            cols = base_col + local_cols[nonzero_idx]
            vals = block_flat[nonzero_idx]
            
            rows_list.append(rows)
            cols_list.append(cols)
            data_list.append(vals)
    
    # 合并所有数据
    if rows_list:
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        data = np.concatenate(data_list)
    else:
        rows, cols, data = np.array([]), np.array([]), np.array([])
    
    return rows, cols, data, (n_dof, n_dof)  