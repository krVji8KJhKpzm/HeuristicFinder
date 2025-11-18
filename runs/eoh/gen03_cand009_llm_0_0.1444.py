# score=0.144422
# gamma=-0.100000
# code_hash=948da801474913ef9ed2db71f330b186f78363384b494b85e41c1da0b92ac1a8
# stats: mse=1.8778; rmse=1.37033; mse_tsp100=6.92416; mse_tsp20=1.8778; mse_tsp50=3.85673; mse_worst=6.92416; rmse_tsp100=2.63138; rmse_tsp20=1.37033; rmse_tsp50=1.96386; rmse_worst=2.63138
# ALGORITHM: {auto} def phi(state): """ Estimates the future tour length (cost-to-go) for the TSP environment. {The algorithm estimates the remaining tour length by multiplying the number of unvisited nodes by the average minimum distance from each unvisited node to any other unvisited node.}
# THOUGHT: {auto}
def phi(state):
    """
    Estimates the future tour length (cost-to-go) for the TSP environment.
    {The algorithm estimates the remaining tour length by multiplying the number of unvisited nodes by the average minimum distance from each unvisited node to any other unvisited node.}

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
    n_unvisited = unvisited_mask.sum(dim=1, dtype=torch.float32)

    # For terminal states, the future cost is zero.
    is_not_terminal = n_unvisited > 1
    if not is_not_terminal.any():
        return torch.zeros(dist_matrix.size(0), 1, device=dist_matrix.device)

    # A large value to mask out irrelevant distances
    large_value = dist_matrix.max() * 2 + 1 if dist_matrix.numel() > 0 else 1e9

    # Create a temporary distance matrix where distances to/from visited nodes are masked
    # [B, N, N]
    temp_dist_matrix = dist_matrix.clone()
    # Mask rows and columns corresponding to visited nodes
    visited_mask = ~unvisited_mask
    temp_dist_matrix[visited_mask, :] = large_value
    temp_dist_matrix[:, visited_mask] = large_value
    # Mask self-loops
    temp_dist_matrix.diagonal(dim1=-2, dim2=-1).fill_(large_value)

    # Find the minimum distance from each unvisited node to another unvisited node
    # [B, N]
    min_dists, _ = temp_dist_matrix.min(dim=2)

    # Set distances from visited nodes to 0 so they don't affect the sum
    min_dists.masked_fill_(visited_mask, 0)

    # Calculate the average of these minimum distances
    # Use a safe divisor to avoid division by zero for terminal states
    safe_n_unvisited = torch.where(n_unvisited > 0, n_unvisited, torch.ones_like(n_unvisited))
    # [B]
    avg_min_dist = min_dists.sum(dim=1) / safe_n_unvisited

    # Estimate the remaining tour length
    # [B]
    remaining_tour_length = n_unvisited * avg_min_dist

    # Ensure terminal states have zero value
    value = torch.where(is_not_terminal, remaining_tour_length, torch.tensor(0.0, device=dist_matrix.device))

    return value.unsqueeze(-1)