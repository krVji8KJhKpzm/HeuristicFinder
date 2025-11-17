# score=0.168591
# gamma=-0.100000
# code_hash=c0011dfdd305034a0384e65a10dd8a1d5dee9f227e2038e32c18321acaf3c85d
# stats: mse=5.93151; rmse=2.43547; mse_tsp100=258.231; mse_tsp20=5.93151; mse_tsp50=55.6125; rmse_tsp100=16.0696; rmse_tsp20=2.43547; rmse_tsp50=7.45738
# ALGORITHM: {auto} def phi(state): """ {My algorithm estimates the future tour length by combining the average distance from the current node to all unvisited nodes with the average pairwise distance among all unvisited nodes, scaled by the number of remaining nodes.}
# THOUGHT: {auto}
def phi(state):
    """
    {My algorithm estimates the future tour length by combining the average distance from the current node to all unvisited nodes with the average pairwise distance among all unvisited nodes, scaled by the number of remaining nodes.}
    Args:
        state: A TSPStateView object with batch-friendly helpers.
    Returns:
        value: A tensor of shape [B, 1] representing the estimated future tour length.
    """
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    # [B]
    current_node = state.current_node_index()
    # [B, 1]
    num_unvisited = unvisited_mask.sum(dim=1, keepdim=True).float()

    # Create a mask for visited nodes to handle the terminal state gracefully
    # [B, N]
    visited_mask = ~unvisited_mask
    # [B, 1]
    is_terminal = (num_unvisited == 0)

    # 1. Average distance from the current node to all unvisited nodes
    # [B, 1, N]
    current_node_exp = current_node.unsqueeze(1).unsqueeze(2).expand(-1, 1, dist_matrix.shape[2])
    # [B, N]
    dist_from_current = dist_matrix.gather(1, current_node_exp).squeeze(1)
    
    # Apply mask and sum
    # [B]
    sum_dist_from_current_to_unvisited = (dist_from_current * unvisited_mask).sum(dim=1)
    # [B, 1]
    avg_dist_from_current_to_unvisited = sum_dist_from_current_to_unvisited.unsqueeze(1) / torch.max(torch.ones_like(num_unvisited), num_unvisited)

    # 2. Average pairwise distance among all unvisited nodes
    # [B, N, N]
    unvisited_pairwise_mask = unvisited_mask.unsqueeze(2) * unvisited_mask.unsqueeze(1)
    # Set diagonal to False to avoid self-distances
    unvisited_pairwise_mask.diagonal(dim1=-2, dim2=-1).fill_(False)
    
    # [B]
    sum_dist_unvisited = (dist_matrix * unvisited_pairwise_mask).sum(dim=[1, 2])
    # [B, 1]
    num_pairs = num_unvisited * (num_unvisited - 1)
    # [B, 1]
    avg_dist_unvisited = sum_dist_unvisited.unsqueeze(1) / torch.max(torch.ones_like(num_pairs), num_pairs)

    # 3. Combine heuristics
    # The expected length of a path through N points is roughly (N-1) * avg_edge_length.
    # Here, we use a combination of the 'entry cost' (from current to the unvisited set)
    # and the 'internal cost' (path within the unvisited set).
    # A simple heuristic is: cost_to_enter + (num_unvisited - 1) * avg_internal_dist
    # Using average distances helps with node-count invariance and scaling.
    # We use num_unvisited as a scaling factor.
    estimated_future_length = avg_dist_from_current_to_unvisited + (num_unvisited - 1).clamp(min=0) * avg_dist_unvisited

    # Get coordinates to calculate a bounding box diagonal as a scaling factor
    # [B, N, 2]
    coords = state.all_node_coords()
    # [B, 2]
    min_coords, _ = coords.min(dim=1)
    max_coords, _ = coords.max(dim=1)
    # [B, 1]
    diag = (max_coords - min_coords).norm(dim=1, keepdim=True)
    
    # Scale the estimate by the bounding box diagonal to normalize across different map scales
    # Add a small epsilon to diag to prevent division by zero for single-node cases
    scaled_estimate = estimated_future_length / (diag + 1e-6)

    # For terminal states (all nodes visited), the future length is 0.
    # This ensures Phi(s_T) = 0.
    value = torch.where(is_terminal, torch.zeros_like(scaled_estimate), scaled_estimate)

    return value