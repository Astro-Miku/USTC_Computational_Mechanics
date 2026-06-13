from B_calcu_metrix import getD,LN,N_ijk
from F_find_tri import find_triangle_of_point
from G_get_displacement import load_result
g,U = load_result('case/grid.npz','case/DisplacementResult.npz')
tri=find_triangle_of_point(g, 30, 1.5)
i1, i2, i3, *_ = tri
v1,v2,v3=g.vertices[i1],g.vertices[i2],g.vertices[i3]
E=200e9
niu=0.3
DB=getD(E,niu,is_stress=1)@LN(N_ijk(v1,v2,v3))
tri_Displace=[U[2 * i1],U[2 * i1 + 1],U[2 * i2],U[2 * i2 + 1],U[2 * i3],U[2 * i3+1]]
[sigma_x,sigma_y,tou_xy]=DB @ tri_Displace
print(sigma_x,sigma_y,tou_xy)
Mises = (sigma_x**2 + sigma_y**2 - sigma_y*sigma_x +3*tou_xy*tou_xy)**0.5
print(Mises)