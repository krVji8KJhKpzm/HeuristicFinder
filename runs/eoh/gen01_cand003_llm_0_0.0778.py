# score=0.077827
# gamma=-0.100000
# code_hash=5fd3309104022111fb964be45f064a888499b1e8ed8f66c21cadfee07bffcee5
# stats: mse=1.58737; rmse=1.25991; mse_tsp100=12.849; mse_tsp20=1.58737; mse_tsp50=4.67237; mse_worst=12.849; rmse_tsp100=3.58455; rmse_tsp20=1.25991; rmse_tsp50=2.16157; rmse_worst=3.58455
# ALGORITHM: {Estimate the future tour length by summing the average distance from the current node to all unvisited nodes and the average distance from the start node to all unvisited nodes.}
# THOUGHT: {Estimate the future tour length by summing the average distance from the current node to all unvisited nodes and the average distance from the start node to all unvisited nodes.}
def phi(state):
    """
    Estimates future tour length by averaging connection costs to the unvisited set.
    The value is the sum of two components:
    1. The average distance from the current node to all unvisited nodes.
    2. The average distance from the start node to all unvisited nodes (to close the tour).
    This heuristic focuses on the expected cost of the next and final connections.

    Args:
        state (TSPStateView): The current state of the TSP environment.
    Returns:
        torch.Tensor: A scalar tensor [B, 1] representing the estimated future cost.
    """
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    B, N = unvisited_mask.shape
    device = unvisited_mask.device
    
    # [B, 1]
    num_unvisited = unvisited_mask.sum(dim=1, keepdim=True)
    is_done = (num_unvisited == 0)

    # Use a small epsilon to avoid division by zero when no nodes are unvisited
    # The result for this case will be masked out later anyway.
    num_unvisited_safe = num_unvisited.clamp(min=1.0)

    # [B, N, N]
    dist_matrix = state.distance_matrix()

    # 1. Average distance from the current node to the unvisited set
    # [B, 1, N]
    current_node_idx = state.current_node_index().view(B, 1, 1).expand(-1, 1, N)
    # [B, N]
    dist_from_current = torch.gather(dist_matrix, 1, current_node_idx).squeeze(1)
    
    # Mask out distances to already visited nodes
    dist_from_current_unvisited = dist_from_current.masked_fill(~unvisited_mask, 0.0)
    
    # [B, 1]
    sum_dist_from_current = dist_from_current_unvisited.sum(dim=1, keepdim=True)
    avg_dist_from_current = sum_dist_from_current / num_unvisited_safe

    # 2. Average distance from the start node to the unvisited set (for closing the loop)
    # [B, 1, N]
    start_node_idx = state.first_node_index().view(B, 1, 1).expand(-1, 1, N)
    # [B, N]
    dist_from_start = torch.gather(dist_matrix, 1, start_node_idx).squeeze(1)

    # Mask out distances to already visited nodes
    dist_from_start_unvisited = dist_from_start.masked_fill(~unvisited_mask, 0.0)

    # [B, 1]
    sum_dist_from_start = dist_from_start_unvisited.sum(dim=1, keepdim=True)
    avg_dist_from_start = sum_dist_from_start / num_unvisited_safe

    # Combine the components
    # [B, 1]
    value = avg_dist_from_current + avg_dist_from_start
    
    # Handle the special case where only one unvisited node remains.
    # The cost is exactly current -> last -> start.
    is_last_step = (num_unvisited == 1)
    if torch.any(is_last_step):
        # [B_last, 1]
        last_unvisited_idx = unvisited_mask[is_last_step.squeeze(-1)].long().argmax(dim=1, keepdim=True)
        # [B_last]
        current_node_last = state.current_node_index()[is_last_step.squeeze(-1)]
        # [B_last]
        start_node_last = state.first_node_index()[is_last_step.squeeze(-1)]
        
        # [B_last, N, N]
        dist_matrix_last = dist_matrix[is_last_step.squeeze(-1)]
        
        # [B_last]
        dist_curr_to_last = torch.gather(dist_matrix_last[:, current_node_last], 1, last_unvisited_idx).squeeze(-1)
        # [B_last]
        dist_last_to_start = torch.gather(dist_matrix_last[:, start_node_last], 1, last_unvisited_idx).squeeze(-1)
        
        last_step_cost = (dist_curr_to_last + dist_last_to_start).unsqueeze(-1)
        value = torch.where(is_last_step, last_step_cost, value)

    # If the tour is done, the future cost is zero.
    value.masked_fill_(is_done, 0.0)

    return value