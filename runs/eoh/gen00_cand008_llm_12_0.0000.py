# score=0.000043
# gamma=0.100000
# code_hash=110b52ce4be499b2e570598accb70428a11c0a3e4c7a503ec60bdbf0b645105b
# stats: mse=399.872; rmse=19.9968; mse_tsp100=23503.6; mse_tsp20=399.872; mse_tsp50=4516.39; mse_worst=23503.6; rmse_tsp100=153.309; rmse_tsp20=19.9968; rmse_tsp50=67.2041; rmse_worst=153.309
# ALGORITHM: {auto} def phi(state): """ Estimates the future tour length (cost-to-go) for the TSP environment. {My algorithm approximates the future tour length by summing two components: the average distance from the current node to all unvisited nodes, and the expected length of the remaining sub-tour among unvisited nodes, which is estimated as the product of the remaining number of edges and the average inter-node distance within the unvisited set.}
# THOUGHT: {auto}
def phi(state):
    """
    Estimates the future tour length (cost-to-go) for the TSP environment.
    {My algorithm approximates the future tour length by summing two components: the average distance from the current node to all unvisited nodes, and the expected length of the remaining sub-tour among unvisited nodes, which is estimated as the product of the remaining number of edges and the average inter-node distance within the unvisited set.}
    """
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    # [B, 1]
    num_unvisited = unvisited_mask.float().sum(dim=1, keepdim=True)
    # [B] -> [B, 1, 1] for broadcasting
    current_node_idx = state.current_node_index().view(-1, 1, 1)

    # 1. Cost from current node to the next unvisited node
    # [B, 1, N]
    dists_from_current = torch.gather(dist_matrix, 1, current_node_idx.expand(-1, 1, dist_matrix.size(2)))
    # [B, 1, N] -> [B, N]
    dists_from_current = dists_from_current.squeeze(1)
    # Zero out distances to visited nodes
    dists_to_unvisited = dists_from_current * unvisited_mask.float()
    # Sum of distances to unvisited nodes [B, 1]
    sum_dists_to_unvisited = dists_to_unvisited.sum(dim=1, keepdim=True)
    # Average distance to an unvisited node [B, 1]
    # Add a small epsilon to avoid division by zero when num_unvisited is 0
    avg_dist_to_next = sum_dists_to_unvisited / torch.clamp(num_unvisited, min=1.0)

    # 2. Cost of the remaining sub-tour among unvisited nodes
    # [B, N, N] -> [B, N, N], mask rows
    unvisited_rows = dist_matrix * unvisited_mask.float().unsqueeze(2)
    # [B, N, N] -> [B, N, N], mask columns
    unvisited_submatrix = unvisited_rows * unvisited_mask.float().unsqueeze(1)
    # Sum of all distances in the submatrix [B, 1]
    sum_inter_unvisited_dists = unvisited_submatrix.sum(dim=[1, 2], keepdim=True)
    # Number of pairs in the unvisited set: N_unvisited * (N_unvisited - 1)
    num_pairs = num_unvisited * (num_unvisited - 1.0)
    # Average distance between any two unvisited nodes [B, 1]
    # Add epsilon to avoid division by zero
    avg_inter_unvisited_dist = sum_inter_unvisited_dists / torch.clamp(num_pairs, min=1.0)

    # The remaining tour length is roughly (num_unvisited - 1) edges.
    # We use num_unvisited as a simple approximation for the number of edges.
    remaining_subtour_len = avg_inter_unvisited_dist * torch.clamp(num_unvisited - 1.0, min=0.0)

    # Total estimated future cost
    # For a completed tour, num_unvisited is 0, so both terms become 0.
    value = avg_dist_to_next + remaining_subtour_len

    # The potential should be negative because we are minimizing length (maximizing negative length)
    return -value