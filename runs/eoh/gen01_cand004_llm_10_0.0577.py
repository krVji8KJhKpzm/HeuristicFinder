# score=0.057697
# gamma=0.100000
# code_hash=df7489edeace72aa540d26fe77201af647d81f86388de9934a7284bd97b03348
# stats: mse=4.03599; rmse=2.00898; mse_tsp100=17.332; mse_tsp20=4.03599; mse_tsp50=8.99578; mse_worst=17.332; rmse_tsp100=4.16317; rmse_tsp20=2.00898; rmse_tsp50=2.9993; rmse_worst=4.16317
# ALGORITHM: {auto} def phi(state): """ {Estimate future tour length by summing the minimum connection cost for each unvisited node and adding the cost to return to the start.}
# THOUGHT: {auto}
def phi(state):
    """
    {Estimate future tour length by summing the minimum connection cost for each unvisited node and adding the cost to return to the start.}
    Args:
        state: A view of the current state of the TSP environment.
    Returns:
        value: A tensor of shape [B, 1] representing the estimated future tour length.
    """
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    # [B] -> [B, 1, 1] -> [B, 1, N]
    current_node = state.current_node_index().unsqueeze(-1).unsqueeze(-1).expand(-1, 1, dist_matrix.size(1))
    # [B, 1, N]
    current_distances = torch.gather(dist_matrix, 1, current_node)
    # [B, N]
    current_distances_squeezed = current_distances.squeeze(1)

    # [B] -> [B, 1]
    first_node = state.first_node_index().unsqueeze(-1)
    # [B, 1]
    dist_to_start = torch.gather(current_distances_squeezed, 1, first_node)

    # For each unvisited node, find the minimum distance to any other node (including current)
    # Create a mask that includes unvisited nodes and the current node as potential connection points
    # [B, N]
    connect_to_mask = unvisited_mask.clone()
    # [B, N]
    connect_to_mask.scatter_(1, state.current_node_index().unsqueeze(-1), True)
    # [B, 1, N]
    connect_to_mask_expanded = connect_to_mask.unsqueeze(1)

    # Mask the distance matrix to only consider connections to valid nodes
    # Set distances to invalid connection points to infinity
    # [B, N, N]
    masked_dist = dist_matrix.clone()
    masked_dist[~connect_to_mask_expanded.expand_as(dist_matrix)] = torch.finfo(dist_matrix.dtype).max

    # Find the minimum distance from each node to a valid connection point
    # [B, N]
    min_dist_to_connect, _ = torch.min(masked_dist, dim=2)

    # Sum these minimum distances only for the unvisited nodes
    # [B, N]
    min_dist_to_connect.masked_fill_(~unvisited_mask, 0)
    # [B]
    sum_min_dists = torch.sum(min_dist_to_connect, dim=1)

    # The total estimated cost is the sum of minimum connection costs for unvisited nodes
    # plus the cost to return to the start node from the current node.
    # [B]
    value = sum_min_dists + dist_to_start.squeeze(1)

    return value.unsqueeze(-1)