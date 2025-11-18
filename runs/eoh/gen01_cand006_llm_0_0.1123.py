# score=0.112306
# gamma=-0.100000
# code_hash=0e7601ed649dfde68fc9175671d4df815cf1f6e8088ee79de00b3b2d9f6dd931
# stats: mse=0.90906; rmse=0.953446; mse_tsp100=8.90422; mse_tsp20=0.90906; mse_tsp50=3.43223; mse_worst=8.90422; rmse_tsp100=2.98399; rmse_tsp20=0.953446; rmse_tsp50=1.85263; rmse_worst=2.98399
# ALGORITHM: {Estimate the future tour length by calculating the area of the convex hull of unvisited nodes and adding the cost to connect the current node and the starting node to this hull.}
# THOUGHT: {Estimate the future tour length by calculating the area of the convex hull of unvisited nodes and adding the cost to connect the current node and the starting node to this hull.}
def phi(state):
    """
    Estimates future tour length based on the geometric properties of unvisited nodes.
    The value is composed of three parts:
    1. An approximation of the sub-tour length for unvisited nodes using the area of their convex hull.
    2. The minimum distance from the current node to any of the unvisited nodes.
    3. The minimum distance from the start node to any of the unvisited nodes.
    This combines a geometric heuristic for the bulk of the path with connection costs.

    Args:
        state (TSPStateView): The current state of the TSP environment.
    Returns:
        torch.Tensor: A scalar tensor [B, 1] representing the estimated future cost.
    """
    # [B, N, 2]
    coords = state.all_node_coords()
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    B, N, _ = coords.shape
    device = coords.device

    # Handle terminal and near-terminal states
    num_unvisited = unvisited_mask.sum(dim=1, keepdim=True)  # [B, 1]
    is_done = (num_unvisited == 0)

    # 1. Approximate sub-tour length using convex hull area of unvisited nodes
    # A heuristic for tour length is related to the square root of the area of the convex hull.
    # We use a simplified proxy: the area of the bounding box of unvisited nodes.

    # Mask coordinates of visited nodes to exclude them from min/max calculations
    # [B, N, 2]
    masked_coords = coords.clone()
    # Use a large value for visited nodes so they don't affect min, and a small value for max
    large_val = torch.finfo(coords.dtype).max
    small_val = torch.finfo(coords.dtype).min
    # [B, N, 1]
    unvisited_mask_exp = unvisited_mask.unsqueeze(-1)
    masked_coords.masked_fill_(~unvisited_mask_exp, large_val)
    min_coords, _ = torch.min(masked_coords, dim=1)  # [B, 2]

    masked_coords = coords.clone()
    masked_coords.masked_fill_(~unvisited_mask_exp, small_val)
    max_coords, _ = torch.max(masked_coords, dim=1)  # [B, 2]

    # Calculate width and height of the bounding box
    # [B, 2]
    span = max_coords - min_coords
    # Clamp to avoid negative span if only one node is left
    span = torch.clamp(span, min=0.0)
    # [B]
    area = span[:, 0] * span[:, 1]
    # Heuristic for tour length: k * sqrt(Area * num_unvisited), k is an empirical factor
    # For a unit square with N points, E[tour_len] ~ beta * sqrt(N), and Area=1.
    # We use a simplified form: sqrt(Area) as a proxy for the spatial scale.
    # The sum of two sides of the bounding box is a reasonable perimeter proxy.
    subtour_cost = span[:, 0] + span[:, 1]

    # 2. Cost to connect the current node to the unvisited set
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, 1, N]
    current_node_idx = state.current_node_index().view(B, 1, 1).expand(-1, 1, N)
    # [B, N]
    dist_from_current = torch.gather(dist_matrix, 1, current_node_idx).squeeze(1)
    dist_from_current.masked_fill_(~unvisited_mask, torch.finfo(dist_from_current.dtype).max)
    # [B]
    min_dist_to_unvisited, _ = torch.min(dist_from_current, dim=1)

    # 3. Cost to connect the start node to the unvisited set (for closing the loop)
    # [B, 1, N]
    start_node_idx = state.first_node_index().view(B, 1, 1).expand(-1, 1, N)
    # [B, N]
    dist_from_start = torch.gather(dist_matrix, 1, start_node_idx).squeeze(1)
    dist_from_start.masked_fill_(~unvisited_mask, torch.finfo(dist_from_start.dtype).max)
    # [B]
    min_dist_from_start, _ = torch.min(dist_from_start, dim=1)

    # Combine the components
    # [B]
    value = subtour_cost + min_dist_to_unvisited + min_dist_from_start
    
    # Handle edge case where only one unvisited node remains
    # The cost is just from current -> last_unvisited -> start
    is_last_step = (num_unvisited.squeeze(-1) == 1)
    if torch.any(is_last_step):
        # [B_last, 1]
        last_unvisited_idx = unvisited_mask[is_last_step].long().argmax(dim=1, keepdim=True)
        # [B_last]
        current_node_last = state.current_node_index()[is_last_step]
        # [B_last]
        start_node_last = state.first_node_index()[is_last_step]

        # [B_last]
        dist_curr_to_last = torch.gather(dist_matrix[is_last_step, current_node_last], 1, last_unvisited_idx).squeeze(-1)
        # [B_last]
        dist_last_to_start = torch.gather(dist_matrix[is_last_step, start_node_last], 1, last_unvisited_idx).squeeze(-1)
        
        last_step_cost = dist_curr_to_last + dist_last_to_start
        value[is_last_step] = last_step_cost

    # If the tour is done, future cost is 0
    value.masked_fill_(is_done.squeeze(-1), 0.0)
    # Replace any remaining infs (e.g., from min over empty set if num_unvisited=0) with 0
    value = torch.nan_to_num(value, posinf=0.0)

    return value.unsqueeze(-1)