# score=0.048128
# gamma=1.000000
# code_hash=50dcd093dd12b92fd0b2636561caf4c7da89b61f13eef279de583aec6eb021ac
# stats: mse=20.7778; rmse=4.55826; mse_tsp100=73.3897; mse_tsp20=20.7778; mse_tsp50=40.7692; rmse_tsp100=8.56678; rmse_tsp20=4.55826; rmse_tsp50=6.38507
# ALGORITHM: {auto}
# THOUGHT: {auto}
def phi(state):
    """
    Estimates the future tour length (negative value) for the TSP environment.
    The value is estimated based on the connectivity of the remaining unvisited nodes
    and the distance to the next likely node.

    Args:
        state (TSPStateView): The current state of the TSP environment.

    Returns:
        torch.Tensor: A scalar potential value for each state in the batch, shape [B, 1].
    """
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    # [B, N, 1]
    unvisited_mask_from = unvisited_mask.unsqueeze(2)
    # [B, 1, N]
    unvisited_mask_to = unvisited_mask.unsqueeze(1)
    # [B, N, N], boolean mask for edges between unvisited nodes
    unvisited_to_unvisited_mask = unvisited_mask_from & unvisited_mask_to

    # Set distances for invalid (visited-involved) connections to a large value
    # so they are ignored by the min operation.
    large_value = 1e9
    # [B, N, N]
    dists_unvisited = torch.where(
        unvisited_to_unvisited_mask,
        dist_matrix,
        torch.full_like(dist_matrix, large_value)
    )

    # For each unvisited node, find the minimum distance to another unvisited node.
    # We set the diagonal to a large value to avoid picking the node itself (dist=0).
    # [B, N]
    diag_mask = torch.eye(dist_matrix.size(1), device=dist_matrix.device).unsqueeze(0)
    dists_unvisited = torch.where(diag_mask, large_value, dists_unvisited)
    # [B, N], min dist from each node `i` to any other unvisited node `j`
    min_dists, _ = torch.min(dists_unvisited, dim=2)

    # Sum these minimum distances only for the unvisited nodes.
    # [B]
    sum_min_dists = torch.sum(min_dists * unvisited_mask, dim=1)

    # Get the number of unvisited nodes. Add a small epsilon to avoid division by zero.
    # [B]
    n_unvisited = torch.sum(unvisited_mask, dim=1).float()
    
    # Calculate the average minimum distance between unvisited nodes.
    # This represents the expected cost per hop in the remainder of the tour.
    # [B]
    avg_min_dist = sum_min_dists / torch.clamp(n_unvisited, min=1.0)
    
    # Estimate the total remaining path length based on this average.
    # [B]
    remaining_path_estimate = avg_min_dist * n_unvisited

    # Add the cost to get from the current node to the nearest unvisited node.
    # [B]
    current_node = state.current_node_index()
    # [B, N]
    dists_from_current = torch.gather(dist_matrix, 1, current_node.view(-1, 1, 1).expand(-1, 1, dist_matrix.size(2))).squeeze(1)
    # [B, N]
    dists_from_current_to_unvisited = torch.where(
        unvisited_mask,
        dists_from_current,
        torch.full_like(dists_from_current, large_value)
    )
    # [B]
    min_dist_to_next, _ = torch.min(dists_from_current_to_unvisited, dim=1)
    
    # If there are no unvisited nodes, this distance is inf. Set it to 0.
    # [B]
    is_done = (n_unvisited == 0)
    min_dist_to_next = torch.where(is_done, 0.0, min_dist_to_next)

    # The final potential is the sum of the estimated remaining path and the immediate next step.
    # The negative sign is because rewards are negative distances.
    # [B]
    value = -(remaining_path_estimate + min_dist_to_next)
    
    # Ensure terminal states (and states with 1 unvisited node, which is the last step) have a potential of 0.
    # This helps with terminal consistency.
    is_terminal_or_near_terminal = (n_unvisited <= 1)
    value = torch.where(is_terminal_or_near_terminal, 0.0, value)

    return value.unsqueeze(1)