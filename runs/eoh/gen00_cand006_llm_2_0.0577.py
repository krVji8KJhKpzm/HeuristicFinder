# score=0.057704
# gamma=-0.100000
# code_hash=c6e9e810e91f1fd5f94481e92b67367e5e737df98b41d0d6f3a71ee61d93e325
# stats: mse=17.3298; rmse=4.16291; mse_tsp100=596.154; mse_tsp20=17.3298; mse_tsp50=137.471; rmse_tsp100=24.4163; rmse_tsp20=4.16291; rmse_tsp50=11.7248
# ALGORITHM: {auto} def phi(state): """ Estimates the future tour length (cost-to-go) for the TSP environment. {The algorithm estimates the future tour length by combining the expected distance to the next node (average distance from current to unvisited) with the estimated cost of the remaining sub-tour (average inter-node distance for unvisited nodes scaled by the number of remaining nodes).}
# THOUGHT: {auto}
def phi(state):
    """
    Estimates the future tour length (cost-to-go) for the TSP environment.
    {The algorithm estimates the future tour length by combining the expected distance to the next node (average distance from current to unvisited) with the estimated cost of the remaining sub-tour (average inter-node distance for unvisited nodes scaled by the number of remaining nodes).}
    """
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    # [B, N, 2]
    coords = state.all_node_coords()
    # [B]
    current_node = state.current_node_index()

    # Number of unvisited nodes, [B]
    num_unvisited = unvisited_mask.sum(dim=1, dtype=torch.float32)

    # Create a mask for terminal states (num_unvisited == 0)
    # [B]
    is_terminal = (num_unvisited == 0)

    # 1. Estimate cost of the remaining sub-tour among unvisited nodes
    # Mask for pairs of unvisited nodes: [B, N, 1] * [B, 1, N] -> [B, N, N]
    unvisited_pair_mask = unvisited_mask.unsqueeze(2) & unvisited_mask.unsqueeze(1)
    # Zero out diagonal to avoid counting self-distances
    unvisited_pair_mask.diagonal(dim1=-2, dim2=-1).fill_(False)

    # Sum of distances between all pairs of unvisited nodes, [B]
    sum_dist_unvisited = (dist_matrix * unvisited_pair_mask).sum(dim=(1, 2)) / 2.0

    # Average distance between unvisited nodes
    # Number of pairs: N_unvisited * (N_unvisited - 1) / 2
    num_pairs = num_unvisited * (num_unvisited - 1) / 2.0
    # Avoid division by zero when num_unvisited <= 1, [B]
    avg_dist_unvisited = torch.nan_to_num(sum_dist_unvisited / (num_pairs + 1e-9))

    # Estimated cost of the sub-tour is (num_unvisited) * avg_dist_unvisited
    # When num_unvisited is 1, this is 0. When 0, also 0.
    subtour_cost = (num_unvisited) * avg_dist_unvisited

    # 2. Estimate cost to connect current tour to the remaining sub-tour
    # [B, N]
    current_idx_expanded = current_node.unsqueeze(1)
    # [B, N]
    dist_from_current = dist_matrix.gather(1, current_idx_expanded.unsqueeze(2).expand(-1, -1, dist_matrix.size(2))).squeeze(1)

    # Sum of distances from current node to all unvisited nodes, [B]
    sum_dist_from_current_to_unvisited = (dist_from_current * unvisited_mask).sum(dim=1)
    # Average distance, avoiding division by zero, [B]
    avg_dist_to_next = torch.nan_to_num(sum_dist_from_current_to_unvisited / (num_unvisited + 1e-9))

    # 3. Combine costs
    # Total estimated future cost, [B]
    future_cost = subtour_cost + avg_dist_to_next

    # For terminal states, the future cost is exactly 0.
    # This handles the case where the tour is complete but not returned to start.
    # The environment reward will handle the final edge cost.
    value = torch.where(is_terminal, torch.zeros_like(future_cost), future_cost)

    return value.unsqueeze(1)