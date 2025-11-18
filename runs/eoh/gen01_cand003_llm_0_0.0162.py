# score=0.016241
# gamma=-0.100000
# code_hash=4fd7044f0a2792e1506701c0b8d639b18640f0383c6f3aa38f4a2283890d4310
# stats: mse=28.4978; rmse=5.33834; mse_tsp100=61.5725; mse_tsp20=28.4978; mse_tsp50=42.6798; mse_worst=61.5725; rmse_tsp100=7.84681; rmse_tsp20=5.33834; rmse_tsp50=6.53298; rmse_worst=7.84681
# ALGORITHM: {Estimate the future tour length by calculating the cost of a convex hull around unvisited nodes, plus the costs to connect the current and start nodes to this hull.}
# THOUGHT: {Estimate the future tour length by calculating the cost of a convex hull around unvisited nodes, plus the costs to connect the current and start nodes to this hull.}
def phi(state):
    """
    Estimates the future tour length (cost-to-go) for the TSP using a convex hull heuristic.
    The value is composed of three parts:
    1. The perimeter of the convex hull of the unvisited nodes, which serves as a lower bound for the sub-tour length.
    2. The minimum distance from the current node to any unvisited node (cost to enter the sub-tour).
    3. The minimum distance from the start node to any unvisited node (cost to return from the sub-tour).

    Args:
        state (TSPStateView): The current state of the TSP environment.

    Returns:
        torch.Tensor: A scalar tensor [B, 1] representing the estimated future cost.
    """
    # [B, N, 2]
    coords = state.all_node_coords()
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    # [B, 1]
    num_unvisited = unvisited_mask.sum(dim=1, keepdim=True)

    # 1. Calculate the convex hull perimeter for unvisited nodes
    # For stability, replace coordinates of visited nodes with a far-away point
    # so they don't interfere with the convex hull of unvisited nodes.
    # A large value like 1e9 should be outside any practical TSP coordinate range.
    large_val = torch.full_like(coords, 1e9)
    # [B, N, 2]
    unvisited_coords = torch.where(unvisited_mask.unsqueeze(-1), coords, large_val)

    # Find the min/max x/y coordinates among unvisited nodes to approximate the hull size
    # This avoids complex hull algorithms and is differentiable.
    # [B, 1]
    min_x = torch.min(unvisited_coords[..., 0], dim=1, keepdim=True).values
    max_x = torch.max(unvisited_coords[..., 0].masked_fill_(~unvisited_mask, -1e9), dim=1, keepdim=True).values
    min_y = torch.min(unvisited_coords[..., 1], dim=1, keepdim=True).values
    max_y = torch.max(unvisited_coords[..., 1].masked_fill_(~unvisited_mask, -1e9), dim=1, keepdim=True).values

    # Approximate perimeter of the bounding box as a proxy for convex hull perimeter
    # [B, 1]
    hull_perimeter = 2 * ((max_x - min_x) + (max_y - min_y))
    # Handle cases with 0 or 1 unvisited nodes where hull is undefined or zero
    hull_perimeter = hull_perimeter.masked_fill(num_unvisited <= 1, 0.0)

    # 2. Calculate connection costs to the unvisited set
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B] -> [B, 1, 1] -> [B, 1, N]
    current_node_idx = state.current_node_index().view(-1, 1, 1).expand(-1, 1, dist_matrix.size(1))
    start_node_idx = state.first_node_index().view(-1, 1, 1).expand(-1, 1, dist_matrix.size(1))

    # [B, N]
    dist_from_current = torch.gather(dist_matrix, 1, current_node_idx).squeeze(1)
    dist_from_start = torch.gather(dist_matrix, 1, start_node_idx).squeeze(1)

    # Find minimum distance to any unvisited node
    # [B, 1]
    min_dist_to_unvisited = torch.min(dist_from_current.masked_fill(~unvisited_mask, float('inf')), dim=1, keepdim=True).values
    min_dist_from_unvisited_to_start = torch.min(dist_from_start.masked_fill(~unvisited_mask, float('inf')), dim=1, keepdim=True).values

    # Handle terminal states where no unvisited nodes exist
    min_dist_to_unvisited = min_dist_to_unvisited.nan_to_num(posinf=0.0)
    min_dist_from_unvisited_to_start = min_dist_from_unvisited_to_start.nan_to_num(posinf=0.0)

    # Total estimated future cost
    value = hull_perimeter + min_dist_to_unvisited + min_dist_from_unvisited_to_start

    # Special case for the last step: cost is simply from current to last, then to start
    is_last_step = (num_unvisited == 1)
    if is_last_step.any():
        # [B, 1]
        last_unvisited_idx = unvisited_mask.long().argmax(dim=1, keepdim=True)
        # [B, 1]
        dist_curr_to_last = torch.gather(dist_from_current, 1, last_unvisited_idx)
        dist_last_to_start = torch.gather(dist_from_start, 1, last_unvisited_idx)
        last_step_cost = dist_curr_to_last + dist_last_to_start
        value = torch.where(is_last_step, last_step_cost, value)

    # If the tour is done, future cost is zero.
    is_done = (num_unvisited == 0)
    value = value.masked_fill(is_done, 0.0)

    return -value