# score=0.254326
# gamma=0.100000
# code_hash=b5692d3078bbaa95c8199d4acf4dfb8313a5b628da94d9fd8ac264f2af08cf33
# stats: mse=0.343209; rmse=0.585841; mse_tsp100=3.93197; mse_tsp20=0.343209; mse_tsp50=1.33105; mse_worst=3.93197; rmse_tsp100=1.98292; rmse_tsp20=0.585841; rmse_tsp50=1.15371; rmse_worst=1.98292
# ALGORITHM: {Estimate future tour length by calculating the convex hull area of unvisited nodes and scaling it by the average node density, adding costs to connect the current and start nodes to this hull.} def phi(state): """ Estimates future tour length using a convex hull-based geometric heuristic. The cost is the sum of three components: 1. An estimate of the sub-tour length for unvisited nodes, derived from the area of their convex hull. The intuition is that tour length scales with the square root of the area covered. This is scaled by node density. 2. The minimum cost to connect the current node to the convex hull. 3. The minimum cost to connect the start node to the convex hull, for the return trip. """ coords = state.all_node_coords() unvisited_mask = state.unvisited_mask() B, N, _ = coords.shape device = coords.device n_unvisited = unvisited_mask.sum(dim=1) is_terminal = (n_unvisited == 0) has_enough_nodes = (n_unvisited >= 3) # 1. Estimate sub-tour length for unvisited nodes using convex hull. # Create a copy of coords and mask out visited nodes for hull calculation. unvisited_coords = coords.clone() # Set visited nodes to a far-away point so they don't affect the hull. # A single point (e.g., origin) could cause issues if it's inside the hull. large_val = coords.abs().max() + 1.0 unvisited_coords.masked_fill_(~unvisited_mask.unsqueeze(-1), large_val) # Find the bottom-most, then left-most point to start Graham scan. # Using y-coord then x-coord as tie-breaker. sort_keys = torch.stack([unvisited_coords[..., 1], unvisited_coords[..., 0]], dim=-1) # A large value for masked nodes ensures they are sorted last. sort_keys.masked_fill_(~unvisited_mask.unsqueeze(-1), float('inf')) # Find the index of the start point for each batch item. start_pt_idx = torch.lexsort(dims=-1, keys=sort_keys.permute(2, 0, 1))[0] # Get coordinates of the start point. p0 = torch.gather(coords, 1, start_pt_idx.view(B, 1, 1).expand(-1, -1, 2)) # Calculate angles of all other points with respect to p0. # Subtract p0 to center the coordinate system. centered_coords = coords - p0 # Use atan2 for stable angle calculation. angles = torch.atan2(centered_coords[..., 1], centered_coords[..., 0]) # Mask out visited nodes and the start point itself. angles.masked_fill_(~unvisited_mask, float('inf')) angles.scatter_(1, start_pt_idx.unsqueeze(1), float('inf')) # Sort points by angle to get the hull vertices in order. # The sorted indices give the order of vertices in the convex hull. _, sorted_indices = torch.sort(angles, dim=1) # Gather the coordinates of the sorted unvisited points. sorted_unvisited_coords = torch.gather(coords, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, 2)) # Prepend the starting point to form a closed polygon path. # Path: p0 -> sorted_p1 -> sorted_p2 ... -> p0 hull_path_coords = torch.cat([p0, sorted_unvisited_coords], dim=1) # Calculate polygon area using the shoelace formula. x, y = hull_path_coords[..., 0], hull_path_coords[..., 1] # This computes sum(x_i * y_{i+1} - x_{i+1}
# THOUGHT: {Estimate future tour length by calculating the convex hull area of unvisited nodes and scaling it by the average node density, adding costs to connect the current and start nodes to this hull.}
def phi(state):
    """
    Estimates future tour length using a convex hull-based geometric heuristic.
    The cost is the sum of three components:
    1. An estimate of the sub-tour length for unvisited nodes, derived from the
       area of their convex hull. The intuition is that tour length scales with
       the square root of the area covered. This is scaled by node density.
    2. The minimum cost to connect the current node to the convex hull.
    3. The minimum cost to connect the start node to the convex hull, for the return trip.
    """
    coords = state.all_node_coords()
    unvisited_mask = state.unvisited_mask()
    B, N, _ = coords.shape
    device = coords.device

    n_unvisited = unvisited_mask.sum(dim=1)
    is_terminal = (n_unvisited == 0)
    has_enough_nodes = (n_unvisited >= 3)

    # 1. Estimate sub-tour length for unvisited nodes using convex hull.
    # Create a copy of coords and mask out visited nodes for hull calculation.
    unvisited_coords = coords.clone()
    # Set visited nodes to a far-away point so they don't affect the hull.
    # A single point (e.g., origin) could cause issues if it's inside the hull.
    large_val = coords.abs().max() + 1.0
    unvisited_coords.masked_fill_(~unvisited_mask.unsqueeze(-1), large_val)

    # Find the bottom-most, then left-most point to start Graham scan.
    # Using y-coord then x-coord as tie-breaker.
    sort_keys = torch.stack([unvisited_coords[..., 1], unvisited_coords[..., 0]], dim=-1)
    # A large value for masked nodes ensures they are sorted last.
    sort_keys.masked_fill_(~unvisited_mask.unsqueeze(-1), float('inf'))
    # Find the index of the start point for each batch item.
    start_pt_idx = torch.lexsort(dims=-1, keys=sort_keys.permute(2, 0, 1))[0]

    # Get coordinates of the start point.
    p0 = torch.gather(coords, 1, start_pt_idx.view(B, 1, 1).expand(-1, -1, 2))

    # Calculate angles of all other points with respect to p0.
    # Subtract p0 to center the coordinate system.
    centered_coords = coords - p0
    # Use atan2 for stable angle calculation.
    angles = torch.atan2(centered_coords[..., 1], centered_coords[..., 0])
    # Mask out visited nodes and the start point itself.
    angles.masked_fill_(~unvisited_mask, float('inf'))
    angles.scatter_(1, start_pt_idx.unsqueeze(1), float('inf'))

    # Sort points by angle to get the hull vertices in order.
    # The sorted indices give the order of vertices in the convex hull.
    _, sorted_indices = torch.sort(angles, dim=1)

    # Gather the coordinates of the sorted unvisited points.
    sorted_unvisited_coords = torch.gather(coords, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, 2))
    # Prepend the starting point to form a closed polygon path.
    # Path: p0 -> sorted_p1 -> sorted_p2 ... -> p0
    hull_path_coords = torch.cat([p0, sorted_unvisited_coords], dim=1)

    # Calculate polygon area using the shoelace formula.
    x, y = hull_path_coords[..., 0], hull_path_coords[..., 1]
    # This computes sum(x_i * y_{i+1} - x_{i+1} * y_i)
    area = 0.5 * torch.abs(
        torch.sum(x[:, :-1] * y[:, 1:] - x[:, 1:] * y[:, :-1], dim=1) +
        (x[:, -1] * y[:, 0] - x[:, 0] * y[:, -1])
    )

    # Held's heuristic: length approx. C * sqrt(N * Area)
    # We use a simplified version: sqrt(Area) * sqrt(N_unvisited)
    # The constant factor is learned implicitly by the value function network.
    subtour_len_est = torch.sqrt(area) * torch.sqrt(n_unvisited.float())
    subtour_len_est.masked_fill_(~has_enough_nodes, 0.0)

    # 2. & 3. Cost to connect current and start nodes to the unvisited set.
    # This is a simpler and more robust connection cost than connecting to the hull itself.
    dist_matrix = state.distance_matrix()
    current_node = state.current_node_index()
    start_node = state.first_node_index()

    dist_from_current = torch.gather(dist_matrix, 1, current_node.view(B, 1, 1).expand(-1, -1, N)).squeeze(1)
    dist_from_current.masked_fill_(~unvisited_mask, float('inf'))
    min_dist_to_unvisited, _ = torch.min(dist_from_current, dim=1)

    dist_from_start = torch.gather(dist_matrix, 1, start_node.view(B, 1, 1).expand(-1, -1, N)).squeeze(1)
    dist_from_start.masked_fill_(~unvisited_mask, float('inf'))
    min_dist_from_start, _ = torch.min(dist_from_start, dim=1)

    # Handle cases with fewer than 3 unvisited nodes separately.
    # If 1 or 2 unvisited, path is fixed. Calculate exact future cost.
    # [B, N]
    dist_curr_unvisited = dist_from_current.clone()
    dist_curr_unvisited[dist_curr_unvisited == float('inf')] = -1
    # [B, k] where k is max number of unvisited nodes
    top_two_dists, top_two_indices = torch.topk(dist_curr_unvisited, k=2, dim=1)

    # Cost for 1 unvisited node: dist(current -> u1) + dist(u1 -> start)
    u1_idx = top_two_indices[:, 0]
    dist_u1_start = torch.gather(dist_from_start, 1, u1_idx.unsqueeze(1)).squeeze(1)
    cost_one_unvisited = top_two_dists[:, 0] + dist_u1_start

    # Cost for 2 unvisited nodes: min(path1, path2)
    # path1: current -> u1 -> u2 -> start
    # path2: current -> u2 -> u1 -> start
    u2_idx = top_two_indices[:, 1]
    dist_u2_start = torch.gather(dist_from_start, 1, u2_idx.unsqueeze(1)).squeeze(1)
    dist_u1_u2 = torch.gather(torch.gather(dist_matrix, 1, u1_idx.view(B, 1, 1).expand(-1, -1, N)), 2, u2_idx.view(B, 1, 1)).squeeze()
    cost_path1 = top_two_dists[:, 0] + dist_u1_u2 + dist_u2_start
    cost_path2 = top_two_dists[:, 1] + dist_u1_u2 + dist_u1_start
    cost_two_unvisited = torch.min(cost_path1, cost_path2)

    # Combine all costs
    connection_cost = min_dist_to_unvisited + min_dist_from_start
    value = subtour_len_est + connection_cost

    # Apply exact costs for small numbers of unvisited nodes.
    value = torch.where(n_unvisited == 2, cost_two_unvisited, value)
    value = torch.where(n_unvisited == 1, cost_one_unvisited, value)

    # For terminal states, future cost is 0.
    value.masked_fill_(is_terminal, 0.0)
    # Replace any remaining infs with 0.
    value = torch.nan_to_num(value, posinf=0.0)

    return value.unsqueeze(-1)