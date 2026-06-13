from B_calcu_metrix import getD,LN,N_ijk
from F_find_tri import find_triangle_of_point
from G_get_displacement import load_result

import numpy as np
g,U = load_result('case/grid.npz','case/DisplacementResult.npz')
E=200e9
niu=0.3
sigma_x_arr=[]
for i in range(100):
    tri=find_triangle_of_point(g, 10, 0+0.03*i)
    i1, i2, i3, *_ = tri
    v1,v2,v3=g.vertices[i1],g.vertices[i2],g.vertices[i3]

    DB=getD(E,niu,is_stress=1)@LN(N_ijk(v1,v2,v3))
    tri_Displace=[U[2 * i1],U[2 * i1 + 1],U[2 * i2],U[2 * i2 + 1],U[2 * i3],U[2 * i3+1]]
    [sigma_x,sigma_y,tou_xy]=DB @ tri_Displace
    sigma_x_arr.append(sigma_x)
np.savez('case/biggrid2.npz',arr=sigma_x_arr)