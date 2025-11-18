# score=0.072667
# gamma=1.000000
# code_hash=884346de00c0f4e4ef075e61c49dd857c93553fc2d00215ba1795fcf9caa8a84
# stats: mse=2.34182; rmse=1.5303; mse_tsp100=13.7615; mse_tsp20=2.34182; mse_tsp50=6.42053; mse_worst=13.7615; rmse_tsp100=3.70965; rmse_tsp20=1.5303; rmse_tsp50=2.53388; rmse_worst=3.70965
# ALGORITHM: {Estimate the future tour length by summing the average distance from the current node to all unvisited nodes and the average distance from the start node to all unvisited nodes.}
# THOUGHT: {Estimate the future tour length by summing the average distance from the current node to all unvisited nodes and the average distance from the start node to all unvisited nodes.}
def phi(state):
    """
    Estimates future tour length based on average distances to the set of unvisited nodes.
    This heuristic considers the expected cost to connect the current path segment back
    to the main cluster of unvisited nodes, and then eventually close the tour from that
    cluster back to the start node. It uses average distance as a robust proxy for
    these connection costs, avoiding reliance on single min/max values which can be noisy.

    Args:
        state (TSPStateView): The current state of the TSP environment.
    Returns:
        torch.Tensor: A scalar tensor [B, 1] representing the estimated future cost.
    """
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    B, N, _ = dist_matrix.shape
    device = dist_matrix.device

    # [B]
    num_unvisited = unvisited_mask.sum(dim=1)
    # Avoid division by zero for terminal states.
    # The final value for terminal states will be masked to 0 anyway.
    num_unvisited_safe = torch.clamp(num_unvisited, min=1.0)

    # 1. Calculate the average distance from the current node to all unvisited nodes.
    # [B, 1, N]
    current_node_idx = state.current_node_index().view(B, 1, 1).expand(-1, 1, N)
    # [B, N]
    dist_from_current = torch.gather(dist_matrix, 1, current_node_idx).squeeze(1)
    # Mask out distances to already visited nodes by setting them to 0.
    dist_from_current_unvisited = dist_from_current * unvisited_mask
    # [B]
    avg_dist_from_current = dist_from_current_unvisited.sum(dim=1) / num_unvisited_safe

    # 2. Calculate the average distance from the start node to all unvisited nodes.
    # [B, 1, N]
    start_node_idx = state.first_node_index().view(B, 1, 1).expand(-1, 1, N)
    # [B, N]
    dist_from_start = torch.gather(dist_matrix, 1, start_node_idx).squeeze(1)
    # Mask out distances to already visited nodes.
    dist_from_start_unvisited = dist_from_start * unvisited_mask
    # [B]
    avg_dist_from_start = dist_from_start_unvisited.sum(dim=1) / num_unvisited_safe

    # 3. Handle the case where only one unvisited node remains.
    # The cost is exactly current -> last_unvisited -> start.
    is_last_step = (num_unvisited == 1)
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
        # For the last step, the average distances are just the exact distances, so this is correct.
        avg_dist_from_current[is_last_step] = dist_curr_to_last
        avg_dist_from_start[is_last_step] = dist_last_to_start

    # Combine the two average distance components.
    # This represents the cost to enter the unvisited cluster and the cost to leave it to finish the tour.
    value = avg_dist_from_current + avg_dist_from_start

    # If the tour is done (num_unvisited is 0), the future cost is 0.
    is_done = (num_unvisited == 0)
    value.masked_fill_(is_done, 0.0)

    # Return broadcastable shape [B, 1]
    return value.unsqueeze(-1)