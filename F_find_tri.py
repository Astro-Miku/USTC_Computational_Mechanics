import math

EPS = 1e-12

# ============================================================
# Geometry utilities
# ============================================================
def point_is_vertex(px, py, vx, vy):
    """Check if point coincides with a mesh vertex"""
    return (px - vx) ** 2 + (py - vy) ** 2 < EPS ** 2


def point_on_segment(px, py, x1, y1, x2, y2):
    """Check if point lies on a line segment (including endpoints)"""
    dx = x2 - x1
    dy = y2 - y1

    seg_len2 = dx * dx + dy * dy
    if seg_len2 < EPS ** 2:
        return False  # degenerate segment (should not exist)

    t = ((px - x1) * dx + (py - y1) * dy) / seg_len2
    if t < -EPS or t > 1.0 + EPS:
        return False

    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    return (px - proj_x) ** 2 + (py - proj_y) ** 2 < EPS ** 2


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """
    Distance from point to line segment.
    Assumes: point is not a vertex and not on the segment (checked beforehand).
    """
    dx = x2 - x1
    dy = y2 - y1

    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))

    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    return math.hypot(px - proj_x, py - proj_y)


def point_in_triangle(px, py, v1, v2, v3):
    """Check if point is inside triangle using barycentric coordinates"""
    x1, y1 = v1.x, v1.y
    x2, y2 = v2.x, v2.y
    x3, y3 = v3.x, v3.y

    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < EPS:
        return False

    w1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
    w2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
    w3 = 1 - w1 - w2

    return (-EPS <= w1 <= 1 + EPS and
            -EPS <= w2 <= 1 + EPS and
            -EPS <= w3 <= 1 + EPS)


# ============================================================
# Core: find which triangle contains a given point
# ============================================================
def find_triangle_of_point(graph, px, py):
    """
    Algorithm:
    1. Check if point is a mesh vertex
    2. Find the nearest vertex
    3. Find the nearest edge among that vertex's neighbors
    4. Among the 1-2 candidate triangles sharing that edge, pick the one
       containing the point (barycentric test)
    """
    # ---- 1. check if point is a vertex ----
    for v in graph.vertices:
        if point_is_vertex(px, py, v.x, v.y):
            for tri in graph.triangles:
                i1, i2, i3, *_ = tri
                if v.idx in (i1, i2, i3):
                    return tri

    # ---- 2. nearest vertex ----
    nearest_v = min(
        graph.vertices,
        key=lambda v: (v.x - px) ** 2 + (v.y - py) ** 2
    )

    # ---- 3. nearest neighboring edge ----
    best_nb = None
    best_dist = float("inf")

    for nb in nearest_v.neighbour:
        if point_on_segment(px, py, nearest_v.x, nearest_v.y, nb.x, nb.y):
            best_nb = nb
            break

        dist = point_to_segment_distance(
            px, py, nearest_v.x, nearest_v.y, nb.x, nb.y
        )
        if dist < best_dist:
            best_dist = dist
            best_nb = nb

    # ---- 4. candidate triangles sharing the edge ----
    vi, vj = nearest_v.idx, best_nb.idx
    candidate_tris = [
        tri for tri in graph.triangles
        if vi in tri[:3] and vj in tri[:3]
    ]

    # ---- 5. pick the one containing the point ----
    for tri in candidate_tris:
        i1, i2, i3, *_ = tri
        v1 = graph.vertices[i1]
        v2 = graph.vertices[i2]
        v3 = graph.vertices[i3]

        if point_in_triangle(px, py, v1, v2, v3):
            return tri

    return None