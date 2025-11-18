# score=0.000800
# gamma=-0.100000
# code_hash=da7fdc9bb032d1f5335892b71ec67a7e33e1be96c889b1bb71d5c01fa854e005
# stats: mse=80.9447; rmse=8.99693; mse_tsp100=1250.34; mse_tsp20=80.9447; mse_tsp50=365.089; mse_worst=1250.34; rmse_tsp100=35.3601; rmse_tsp20=8.99693; rmse_tsp50=19.1073; rmse_worst=35.3601
# ALGORITHM: {auto} def phi(state): """ Estimates the future tour length (cost-to-go) for the TSP environment. {My algorithm estimates the future cost by summing three components: the average distance from the current node to all unvisited nodes, the expected tour length for the remaining unvisited nodes, and the distance from the current node back to the starting node.}
# THOUGHT: {auto}
def phi(state):
    """
    Estimates the future tour length (cost-to-go) for the TSP environment.
    {My algorithm estimates the future cost by summing three components: the average distance from the current node to all unvisited nodes, the expected tour length for the remaining unvisited nodes, and the distance from the current node back to the starting node.}
    Args:
        state (TSPStateView): The current state of the TSP environment.
    Returns:
        torch.Tensor: A scalar potential value for each state in the batch, shape [B, 1].
    """
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    # [B]
    current_node = state.current_node_index()
    # [B]
    start_node = state.first_node_index()
    # [B, N, 2]
    coords = state.all_node_coords()
    
    # Number of unvisited nodes, [B]
    num_unvisited = unvisited_mask.sum(dim=1, dtype=torch.float32)
    
    # Avoid division by zero for terminal states where num_unvisited is 0
    is_not_terminal = num_unvisited > 0
    
    # Component 1: Expected distance to the next node
    # Distance from the current node to all other nodes: [B, N]
    dist_from_current = dist_matrix.gather(1, current_node.view(-1, 1, 1).expand(-1, 1, dist_matrix.size(2))).squeeze(1)
    
    # Masked average distance to unvisited nodes
    masked_dist_from_current = dist_from_current * unvisited_mask
    sum_dist_to_unvisited = masked_dist_from_current.sum(dim=1)
    # [B]
    avg_dist_to_next = torch.zeros_like(sum_dist_to_unvisited)
    avg_dist_to_next[is_not_terminal] = sum_dist_to_unvisited[is_not_terminal] / num_unvisited[is_not_terminal]
    
    # Component 2: Estimated length of the tour through remaining unvisited nodes
    # Use average inter-node distance as a proxy for edge length
    # To make it permutation invariant and stable, calculate avg distance over all nodes
    total_dist_sum = dist_matrix.sum(dim=(1, 2))
    N = dist_matrix.size(1)
    # Average distance between any two distinct nodes
    avg_inter_node_dist = total_dist_sum / (N * (N - 1))
    
    # Estimated remaining tour length is (num_unvisited - 1) edges
    # [B]
    remaining_tour_len = (num_unvisited - 1).clamp(min=0) * avg_inter_node_dist
    
    # Component 3: Estimated cost to return to start from the *last* unvisited node
    # We approximate this by the distance from the *current* node to the start node.
    # This is a simplification but captures the closing cost.
    # [B]
    dist_to_start = dist_matrix.gather(1, current_node.view(-1, 1, 1).expand(-1, 1, N)).squeeze(1).gather(1, start_node.view(-1, 1)).squeeze(1)
    
    # Combine components, only for non-terminal states
    # For terminal states, the future cost is 0.
    value = torch.zeros_like(num_unvisited)
    value[is_not_terminal] = avg_dist_to_next[is_not_terminal] + remaining_tour_len[is_not_terminal] + dist_to_start[is_not_terminal]
    
    # The potential should be negative because we are minimizing tour length (maximizing negative length)
    return -value.unsqueeze(1)