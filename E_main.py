# test_direct_solve.py
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from A_module import Vertex, Graph
from D_give_boundary import apply_displacement_constraints
from Y_get_info import load_grid_from_npz

def solve_five_point():
    # 1. load mesh
    npz_file = "case/grid.npz"
    verts, tris = load_grid_from_npz(npz_file)
    g = Graph(verts)
    g.build(tris)

    # 2. element stiffness matrix
    E, nu = 200e9, 0.3
    BCfile = "case/Boun_Cond.npz"

    K, F = apply_displacement_constraints(
        g, E, nu, BCfile, is_force=1
    )
    U = spsolve(K, F)

    # 3. extract data for plotting
    bc_data = np.load(BCfile, allow_pickle=True)
    kind = bc_data['Kind']
    boundary = bc_data['Boundary']
    
    # extract coordinates and displacements
    x_coords = [v.x for v in g.vertices]
    y_coords = [v.y for v in g.vertices]
    u_disp = [U[2*i] for i in range(len(g.vertices))]
    v_disp = [U[2*i+1] for i in range(len(g.vertices))]
    
    # 4. create plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    """
    # Plot 1: displacement vector field
    ax1 = axes[0]
    for i1, i2, i3, _, _, _ in g.triangles:
        v1, v2, v3 = g.vertices[i1], g.vertices[i2], g.vertices[i3]
        x = [v1.x, v2.x, v3.x, v1.x]
        y = [v1.y, v2.y, v3.y, v1.y]
        ax1.plot(x, y, 'lightgray', linewidth=0.5, zorder=1)
    
    scale = 400
    for i, v in enumerate(g.vertices):
        x, y = v.x, v.y
        ux, uy = U[2*i], U[2*i+1]
        
        kx, ky = kind[i]
        bx, by = boundary[i]
        
        # displacement-constrained nodes (red)
        if kx == 1 and ky == 1:
            ax1.scatter(x, y, color='black', s=3, zorder=5)
        elif kx==1 or ky==1:
            ax1.scatter(x, y, color='red', s=3, zorder=5)
        # force boundary nodes (blue arrows)
        if kx == 2 or ky == 2:
            if abs(bx) > 1e-8 or abs(by) > 1e-8:
                ax1.quiver(
                    x, y,
                    bx, by,
                    angles='xy', scale_units='xy', scale=20,
                    color='blue', width=0.004, zorder=5
                )
            else:
                ax1.scatter(x, y, color='green', s=1, zorder=4)
        
        # draw displacement vectors
        if abs(ux) > 1e-10 or abs(uy) > 1e-10:
            ax1.quiver(
                x, y,
                ux, uy,
                angles='xy', scale_units='xy', scale=scale,
                color='gray', width=0.003, alpha=0.8, zorder=6
            )
    
    ax1.set_aspect('equal')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Displacement Vectors')
    ax1.grid(True, alpha=0.3)

    # ✅ New: expand coordinate range for better visibility
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    ax1.set_xlim(x_min - 0.1*(x_max-x_min), x_max + 0.15*(x_max-x_min))
    ax1.set_ylim(y_min - 0.1*(y_max-y_min), y_max + 0.15*(y_max-y_min))
    """
    # Plot 2: u displacement field (x-direction)
    ax2 = axes[0]
    
    # create triangulation
    triangles = np.array([[t[0], t[1], t[2]] for t in g.triangles])
    triang = tri.Triangulation(x_coords, y_coords, triangles)
    
    # plot u displacement field
    tpc_u = ax2.tripcolor(triang, u_disp, cmap='viridis', shading='gouraud')
    plt.colorbar(tpc_u, ax=ax2, label='u displacement')
    
    ax2.set_aspect('equal')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Displacement u (x-direction)')
    
    # Plot 3: v displacement field (y-direction)
    ax3 = axes[1]
    
    # plot v displacement field
    tpc_v = ax3.tripcolor(triang, v_disp, cmap='viridis', shading='gouraud')
    plt.colorbar(tpc_v, ax=ax3, label='v displacement')
    
    ax3.set_aspect('equal')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Displacement v (y-direction)')
    
    plt.tight_layout()
    plt.savefig("case/result.png", dpi=300, bbox_inches='tight')
    plt.close()

    return U


if __name__ == "__main__":
    U=solve_five_point()
    np.savez('case/DisplacementResult.npz',U=U)