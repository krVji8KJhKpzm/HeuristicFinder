# score=0.030136
# gamma=-0.100000
# code_hash=0c563bef19de3f217d8b84ca5c62a92fac8dbc59fdd898899abbc565e6ea1f92
# stats: mse=12.4018; rmse=3.52162; mse_tsp100=33.1831; mse_tsp20=12.4018; mse_tsp50=23.8362; mse_worst=33.1831; rmse_tsp100=5.76048; rmse_tsp20=3.52162; rmse_tsp50=4.88223; rmse_worst=5.76048
# ALGORITHM: {auto} def phi(state): """ Estimates the future tour length (cost-to-go) for the TSP environment. {The algorithm estimates the remaining tour length by summing two components: the average distance from the current node to all unvisited nodes, and the average minimum distance from each unvisited node to another unvisited node, scaled by the number of remaining nodes to visit.}
# THOUGHT: {auto}
def phi(state):
    """
    Estimates the future tour length (cost-to-go) for the TSP environment.
    {The algorithm estimates the remaining tour length by summing two components: the average distance from the current node to all unvisited nodes, and the average minimum distance from each unvisited node to another unvisited node, scaled by the number of remaining nodes to visit.}

    Args:
        state: A TSPStateView object with batch-friendly helper methods.

    Returns:
        A torch tensor of shape [B, 1] representing the estimated future tour length.
    """
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    # [B]
    current_node = state.current_node_index()
    # [B, N, 2]
    coords = state.all_node_coords()
    # [B, N]
    visited_mask = state.visited_mask()

    # Number of unvisited nodes, [B]
    n_unvisited = unvisited_mask.sum(dim=1, dtype=torch.float32)

    # Create a mask for terminal states where no nodes are unvisited
    is_not_terminal = n_unvisited > 0
    # Create a safe divisor to avoid division by zero for terminal states
    safe_n_unvisited = torch.where(is_not_terminal, n_unvisited, torch.ones_like(n_unvisited))

    # Component 1: Average distance from the current node to all unvisited nodes.
    # [B, N]
    current_to_all_dists = torch.gather(dist_matrix, 1, current_node.view(-1, 1, 1).expand(-1, 1, dist_matrix.size(2))).squeeze(1)
    # [B]
    current_to_unvisited_dists = torch.where(unvisited_mask, current_to_all_dists, torch.tensor(0.0, device=dist_matrix.device))
    # [B]
    avg_dist_to_unvisited = current_to_unvisited_dists.sum(dim=1) / safe_n_unvisited

    # Component 2: Average of the minimum distances from each unvisited node to another unvisited node.
    # This estimates the cost of the next step from any potential unvisited node.
    # Mask out visited nodes in the distance matrix by setting their distances to a large value.
    large_value = dist_matrix.max() + 1.0 if dist_matrix.numel() > 0 else 1e9
    # [B, N, N]
    unvisited_dist_matrix = dist_matrix.clone()
    # Mask rows (from) corresponding to visited nodes
    unvisited_dist_matrix[visited_mask, :] = large_value
    # Mask columns (to) corresponding to visited nodes
    unvisited_dist_matrix[:, visited_mask] = large_value
    # Set diagonal to a large value to ignore self-loops
    unvisited_dist_matrix.diagonal(dim1=-2, dim2=-1).fill_(large_value)

    # [B, N]
    min_dists_from_unvisited, _ = unvisited_dist_matrix.min(dim=2)
    # Set distances from visited nodes to 0 so they don't contribute to the sum
    min_dists_from_unvisited = torch.where(unvisited_mask, min_dists_from_unvisited, torch.tensor(0.0, device=dist_matrix.device))
    # [B]
    avg_min_dist_between_unvisited = min_dists_from_unvisited.sum(dim=1) / safe_n_unvisited

    # Heuristic: The remaining tour length is roughly the number of remaining steps
    # times the average cost of a step.
    # We estimate the average step cost using the average minimum distance between unvisited nodes.
    # We add the cost to get to the next node from the current one.
    # n_unvisited already includes the step back to the start if it's unvisited.
    # [B]
    remaining_tour_length = avg_dist_to_unvisited + (n_unvisited - 1).clamp(min=0) * avg_min_dist_between_unvisited

    # For terminal states, the future cost is zero.
    # [B]
    value = torch.where(is_not_terminal, remaining_tour_length, torch.tensor(0.0, device=dist_matrix.device))

    return value.unsqueeze(-1)