# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Mon Jan 12 17:41:04 2026

# @author: sbahalouei
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from shapely.geometry import Polygon, LineString, Point
# from shapely.ops import split

# def solve_poly_clipping_slope(vertices, a, b):
#     # 1. Create the initial domain
#     poly = Polygon(vertices)
    
#     if not poly.equals(poly.convex_hull):
#         print("Warning: The provided domain is not convex.")

#     # 2. Create the line from y = ax + b
#     # We find the bounding box of the polygon to ensure the line is long enough
#     minx, miny, maxx, maxy = poly.bounds
    
#     # Define two points on the line far outside the bounding box
#     x_start, x_end = minx - 10, maxx + 10
#     y_start = a * x_start + b
#     y_end = a * x_end + b
    
#     line = LineString([(x_start, y_start), (x_end, y_end)])

#     # 3. Determine intersection
#     if not line.intersects(poly):
#         return None, None, None

#     # Get intersection points
#     intersection_result = line.intersection(poly.boundary)
#     if isinstance(intersection_result, Point):
#         inter_pts = [intersection_result.coords[0]]
#     else:
#         # Check if it's a MultiPoint or other geometry
#         if hasattr(intersection_result, 'geoms'):
#             inter_pts = [p.coords[0] for p in intersection_result.geoms]
#         else:
#             inter_pts = list(intersection_result.coords)

#     # 4. Split the domain
#     result = split(poly, line)
    
#     # 5. Determine the "Positive Side" (where y > ax + b)
#     positive_domain = None
#     for sub_poly in result.geoms:
#         cx, cy = sub_poly.centroid.x, sub_poly.centroid.y
#         # If the centroid's y is greater than the line's y at that x
#         if cy > (a * cx + b):
#             positive_domain = sub_poly

#     return inter_pts, positive_domain, (x_start, y_start, x_end, y_end)

# # --- Example Usage ---
# # A hexagon-like convex domain
# domain_vertices = [(1, 1), (4, 1), (5, 3), (4, 5), (1, 5), (0, 3), (0, 2)]

# # Line: y = 0.5x + 1.5  (a=0.5, b=1.5)
# slope_a = 0.5
# intercept_b = 1.5

# intersections, new_domain, line_coords = solve_poly_clipping_slope(domain_vertices, slope_a, intercept_b)

# # --- Visualization ---
# if new_domain:
#     fig, ax = plt.subplots(figsize=(7, 7))
    
#     # Original Domain
#     x_orig, y_orig = Polygon(domain_vertices).exterior.xy
#     ax.plot(x_orig, y_orig, color='gray', linestyle='--', label='Original Domain')
    
#     # New "Positive" Domain (y > ax + b)
#     x_new, y_new = new_domain.exterior.xy
#     ax.fill(x_new, y_new, alpha=0.3, color='blue', label='Positive Domain (y > ax+b)')
#     ax.plot(x_new, y_new, color='blue', linewidth=2)
    
#     # The Line
#     x1, y1, x2, y2 = line_coords
#     ax.plot([x1, x2], [y1, y2], color='red', label=f'Line: y={slope_a}x+{intercept_b}')
    
#     # Intersection points
#     ix, iy = zip(*intersections)
#     ax.scatter(ix, iy, color='black', zorder=5)

#     ax.set_xlim(min(x_orig)-1, max(x_orig)+1)
#     ax.set_ylim(min(y_orig)-1, max(y_orig)+1)
#     ax.set_aspect('equal')
#     ax.legend()
#     plt.grid(True, alpha=0.2)
#     plt.show()

#     print("Intersection Points:", intersections)
#     print("New Domain Vertices:", list(new_domain.exterior.coords))






import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import split

def solve_poly_clipping_slope(vertices, a, b, tol=1e-9):
    """
    Clips a convex polygon with the half-plane y > a*x + b.

    Returns:
    - intersection_points (list of tuples)
    - new_domain (Polygon or None)
    - line_coords (x1, y1, x2, y2)
    """

    poly = Polygon(vertices)

    if not poly.is_valid:
        raise ValueError("Invalid polygon.")

    # Bounding box for long line
    minx, miny, maxx, maxy = poly.bounds
    x1, x2 = minx - 10, maxx + 10
    y1, y2 = a * x1 + b, a * x2 + b
    line = LineString([(x1, y1), (x2, y2)])

    # --- CASE 1: No intersection ---
    if not line.intersects(poly):
        cx, cy = poly.centroid.coords[0]
        if cy > a * cx + b + tol:
            # Entire domain is on positive side
            return [], poly, (x1, y1, x2, y2)
        else:
            # Entire domain is on negative side
            return [], None, (x1, y1, x2, y2)

    # --- CASE 2: Intersection exists ---
    inter = line.intersection(poly.boundary)

    if isinstance(inter, Point):
        intersection_points = [inter.coords[0]]
    elif hasattr(inter, "geoms"):
        intersection_points = [p.coords[0] for p in inter.geoms]
    else:
        intersection_points = list(inter.coords)

    # Attempt split
    try:
        pieces = split(poly, line)
    except Exception:
        return intersection_points, None, (x1, y1, x2, y2)

    positive_domain = None
    for piece in pieces.geoms:
        cx, cy = piece.centroid.coords[0]
        if cy > a * cx + b + tol:
            positive_domain = piece
            break

    return intersection_points, positive_domain, (x1, y1, x2, y2)





import matplotlib.pyplot as plt
from shapely.geometry import Polygon

def plot_clipping_result(
    original_vertices,
    intersections,
    new_domain,
    line_coords,
    a,
    b,
    title=None
):
    fig, ax = plt.subplots(figsize=(7, 7))

    # --- Original Domain ---
    poly_orig = Polygon(original_vertices)
    x_orig, y_orig = poly_orig.exterior.xy
    ax.plot(x_orig, y_orig, color='gray', linestyle='--', linewidth=2,
            label='Original Domain')

    # --- New Domain (if exists) ---
    if new_domain is not None:
        x_new, y_new = new_domain.exterior.xy
        ax.fill(x_new, y_new, alpha=0.35, color='blue',
                label='Positive Domain (y > ax + b)')
        ax.plot(x_new, y_new, color='blue', linewidth=2)

    # --- Clipping Line ---
    x1, y1, x2, y2 = line_coords
    ax.plot([x1, x2], [y1, y2],
            color='red', linewidth=2,
            label=f'Line: y = {a}x + {b}')

    # --- Intersection Points ---
    if intersections:
        ix, iy = zip(*intersections)
        ax.scatter(ix, iy, color='black', s=40, zorder=5,
                   label='Intersection Points')

    # --- Axes Settings ---
    ax.set_aspect('equal')
    ax.set_xlim(min(x_orig) - 1, max(x_orig) + 1)
    ax.set_ylim(min(y_orig) - 1, max(y_orig) + 1)

    ax.grid(True, alpha=0.25)
    ax.legend()

    if title:
        ax.set_title(title)

    plt.show()


domain_vertices = [(1,1), (4,1), (5,3), (4,5), (1,5), (0,3)]
a, b = 0.5, 1.5

inter, new_dom, line_coords = solve_poly_clipping_slope(domain_vertices, a, b)

plot_clipping_result(
    domain_vertices,
    inter,
    new_dom,
    line_coords,
    a,
    b,
    title="Intersection Case"
)


a, b = -1.5, 3

inter, new_dom, line_coords = solve_poly_clipping_slope(new_dom, a, b)

plot_clipping_result(
    domain_vertices,
    inter,
    new_dom,
    line_coords,
    a,
    b,
    title="Intersection Case"
)



domain_vertices = [(0,4), (4,4), (4,6), (0,6)]
a, b = 0.5, 1

inter, new_dom, line_coords = solve_poly_clipping_slope(domain_vertices, a, b)

plot_clipping_result(
    domain_vertices,
    inter,
    new_dom,
    line_coords,
    a,
    b,
    title="No Intersection – Domain Fully Positive"
)


domain_vertices = [(0,0), (4,0), (4,1), (0,1)]
a, b = 1, 3

inter, new_dom, line_coords = solve_poly_clipping_slope(domain_vertices, a, b)

plot_clipping_result(
    domain_vertices,
    inter,
    new_dom,
    line_coords,
    a,
    b,
    title="No Intersection – Domain Fully Negative"
)



domain_vertices = [(0, 0), (0, 1), (1, 1), (1,0)]
a, b = -1.77, 0

inter, new_dom, line_coords = solve_poly_clipping_slope(domain_vertices, a, b)

plot_clipping_result(
    domain_vertices,
    inter,
    new_dom,
    line_coords,
    a,
    b,
    title="No Intersection – Domain Fully Negative"
)


domain_vertices = [(0, 0), (0, 1), (1, 1), (1,0)]
a, b = 1.025, 0

inter, new_dom, line_coords = solve_poly_clipping_slope(domain_vertices, a, b)

plot_clipping_result(
    domain_vertices,
    inter,
    new_dom,
    line_coords,
    a,
    b,
    title="No Intersection – Domain Fully Negative"
)
