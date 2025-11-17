# score=0.058655
# gamma=0.100000
# code_hash=96a3268ca8160de2abf7f460fdd4d377097764d1be30329281e8e28d935ecc7a
# stats: mse=17.0488; rmse=4.12902; mse_tsp100=597.18; mse_tsp20=17.0488; mse_tsp50=137.462; rmse_tsp100=24.4373; rmse_tsp20=4.12902; rmse_tsp50=11.7244
# ALGORITHM: {auto}
# THOUGHT: {auto}
def phi(state):
    """
    Estimates the future tour length (cost-to-go) for a TSP state.

    Args:
        state: A TSPStateView object with helper methods.

    Returns:
        A torch tensor of shape [B, 1] representing the estimated future cost.
    """
    # [B, N, 2]
    coords = state.all_node_coords()
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    # [B, 1]
    num_unvisited = unvisited_mask.float().sum(dim=1, keepdim=True)
    # [B]
    current_node = state.current_node_index()
    # [B]
    first_node = state.first_node_index()

    # Create a mask for valid distances to avoid division by zero when no nodes are unvisited
    is_not_done = num_unvisited > 0
    num_unvisited_safe = num_unvisited.clamp(min=1.0)

    # 1. Average distance from the current node to all unvisited nodes.
    # [B, N]
    dist_from_current = torch.gather(dist_matrix, 1, current_node.view(-1, 1, 1).expand(-1, 1, dist_matrix.size(2))).squeeze(1)
    # [B, N] -> [B]
    dist_from_current_to_unvisited = (dist_from_current * unvisited_mask.float()).sum(dim=1)
    # [B]
    avg_dist_to_unvisited = dist_from_current_to_unvisited / num_unvisited_safe.squeeze(1)

    # 2. Average inter-node distance among unvisited nodes, scaled by the remaining tour length.
    # Create a mask for pairs of unvisited nodes: [B, N, 1] * [B, 1, N] -> [B, N, N]
    unvisited_pairs_mask = unvisited_mask.unsqueeze(2) * unvisited_mask.unsqueeze(1)
    # [B, N, N] -> [B]
    sum_inter_unvisited_dist = (dist_matrix * unvisited_pairs_mask.float()).sum(dim=[1, 2])
    # [B]
    num_unvisited_pairs = num_unvisited.squeeze(1) * (num_unvisited.squeeze(1) - 1)
    # [B]
    avg_inter_unvisited_dist = sum_inter_unvisited_dist / num_unvisited_pairs.clamp(min=1.0)
    # [B]
    estimated_unvisited_path_len = avg_inter_unvisited_dist * (num_unvisited.squeeze(1) - 1).clamp(min=0.0)

    # 3. Distance from the current node back to the start node (approximates closing the loop).
    # [B]
    dist_to_start = torch.gather(dist_from_current, 1, first_node.unsqueeze(1)).squeeze(1)

    # Combine the components. The final value is the sum of these estimations.
    # We only consider these costs if the tour is not done.
    value = (avg_dist_to_unvisited + estimated_unvisited_path_len + dist_to_start) * is_not_done.squeeze(1)
    
    # Ensure the output is broadcastable to [B, 1]
    return value.unsqueeze(1)