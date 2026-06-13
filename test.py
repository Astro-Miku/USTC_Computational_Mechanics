from G_get_displacement import load_result , interpolate_displacement
g,U = load_result('case/grid.npz','case/DisplacementResult.npz')
u,v=interpolate_displacement(g, U, 30, 1.5)
print(u,v)