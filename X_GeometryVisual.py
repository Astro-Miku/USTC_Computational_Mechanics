from A_module import Vertex, Edge, Graph
from Y_get_info import load_grid_from_npz
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
npz_file = "case/grid.npz"
verts, tris = load_grid_from_npz(npz_file)
"""
verts = [
        Vertex(0, 0.0, 0.0),
        Vertex(1, 1.0, 0.0),
        Vertex(2, 2.0, 0.0),
        Vertex(3, 0.5, 0.5),
        Vertex(4, 1.5, 0.5),
        Vertex(5, 0.0, 1.0),
        Vertex(6, 1.0, 1.0),
        Vertex(7, 2.0, 1.0),
    ]
tris = [(0,1,3),(1,6,3),(5,6,3),(5,0,3),(1,2,4),(1,6,4),(7,6,4),(7,2,4)]
"""
g = Graph(verts)
g.build(tris)

new_vert, _ = load_grid_from_npz(npz_file)
"""
new_vert = [
        Vertex(0, 0.0, 0.0),
        Vertex(1, 1.0, 0.0),
        Vertex(2, 2.0, 0.0),
        Vertex(3, 0.5, 0.5),
        Vertex(4, 1.5, 0.5),
        Vertex(5, 0.0, 1.0),
        Vertex(6, 1.0, 1.0),
        Vertex(7, 2.0, 1.0),
    ]
"""
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

def extract_all_loops(graph):
    """Extract all boundary loops from the graph."""
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
            
            next_candidates = [n for n in neighbors[current] if n != prev]
            if not next_candidates:
                break
                
            next_idx = next_candidates[0]
            
            if next_idx == idx and len(loop) > 2:
                loop.append(next_idx)
                visited.add(next_idx)
                break
                
            prev = current
            current = next_idx
        
        if len(loop) > 2: 
            loops.append(loop)
    
    return loops

def visualize_and_save(graph, loops):
    """Visualize boundary loops and save data to files."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(loops), 1)))

    # draw each loop
    for i, loop_order in enumerate(loops):
        xs = [graph.vertices[idx].x for idx in loop_order]
        ys = [graph.vertices[idx].y for idx in loop_order]
        xs.append(xs[0])  # close the loop
        ys.append(ys[0])

        # draw the loop
        ax.plot(xs, ys, '-', linewidth=2, marker='o', markersize=6, 
               markerfacecolor=colors[i], label=f'Loop {i}', color=colors[i])
        
        # annotate vertex indices
        for idx in loop_order:
            v = graph.vertices[idx]
            ax.annotate(f'{idx}', (v.x, v.y), 
                       xytext=(3, 3), textcoords='offset points',
                       fontsize=3, fontweight='bold')
        
        # annotate direction (arrows)
        for j in range(len(loop_order)-1):
            x1, y1 = graph.vertices[loop_order[j]].x, graph.vertices[loop_order[j]].y
            x2, y2 = graph.vertices[loop_order[j+1]].x, graph.vertices[loop_order[j+1]].y
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5, alpha=0.7))
    
    # set up plot appearance
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Boundary Loops Visualization')
    ax.grid(True, alpha=0.3)
    
    if loops:
        ax.legend()
    
    # save plot to file
    plt.savefig('case/boundary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # prepare data for saving
    nlink = len(loops)
    head = [loop[0] for loop in loops]  # starting vertex of each loop
    link_data = []

    for loop in loops:
        n = len(loop)
        # only save up to the last element, do not link back to head
        for i in range(n - 1):
            current_idx = loop[i]
            next_idx = loop[i + 1]
            link_data.append([current_idx, next_idx])
    
    # convert to numpy arrays
    head_array = np.array(head)
    link_array = np.array(link_data)
    
    # save to npz file
    np.savez('case/boun_link.npz', 
             nlink=nlink,
             head=head_array,
             link=link_array)
    """
    print(f"Detected {nlink} loops:")
    print(f"\nSaved boundary.png and boun_link.npz")
    print(f"nlink = {nlink}")
    print(f"head = {head_array}")
    print(f"link = {link_array}")
    """
    
    return head_array, link_array

# ========== Execute ==========
loops = extract_all_loops(boun_g)
head_array, link_array = visualize_and_save(boun_g, loops)