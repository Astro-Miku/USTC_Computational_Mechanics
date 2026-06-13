# D_give_boundary.py

import numpy as np
from C_build_whole import calcu_cell, build_whole
from scipy.sparse import coo_matrix
def apply_displacement_constraints(g, E, nu,BCfile,is_force=1):
    data_BC=np.load(BCfile)
    calcu_cell(g, E, nu, is_force)
    rows, cols, data, shape = build_whole(g)
    K = coo_matrix((data, (rows, cols)), shape=shape).tocsr()
    n_dof = shape[0]
    F = np.zeros(n_dof)
    for i in range(n_dof):
        F[i]=data_BC['Boundary'][i//2][i%2]
        if data_BC['Kind'][i//2][i%2]==1:
            K[i,:]=0
            K[i,i]=1
    return K,F