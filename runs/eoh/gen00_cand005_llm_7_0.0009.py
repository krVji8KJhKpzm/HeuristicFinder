# score=0.000912
# gamma=0.100000
# code_hash=47134c722c880296b0d968fcab1d94cb65813f4788c1e5c2dc1acd2812dffb54
# stats: mse=70.7539; rmse=8.41153; mse_tsp100=1097.08; mse_tsp20=70.7539; mse_tsp50=326.047; mse_worst=1097.08; rmse_tsp100=33.1222; rmse_tsp20=8.41153; rmse_tsp50=18.0568; rmse_worst=33.1222
# ALGORITHM: {auto} def phi(state): """ Estimates the future tour length (cost-to-go) for the TSP environment. {The potential function estimates the remaining tour length by combining the average distance from the current node to all unvisited nodes with the expected length of the tour connecting those unvisited nodes, which is approximated by scaling the average inter-node distance by the number of remaining nodes.}
# THOUGHT: {auto}
def phi(state):
    """
    Estimates the future tour length (cost-to-go) for the TSP environment.
    {The potential function estimates the remaining tour length by combining the average distance from the current node to all unvisited nodes with the expected length of the tour connecting those unvisited nodes, which is approximated by scaling the average inter-node distance by the number of remaining nodes.}
    Args:
        state: A TSPStateView object providing access to the environment state.
    Returns:
        A tensor of shape [B, 1] representing the estimated future tour length.
    """
    # [B, N, 2]
    coords = state.all_node_coords()
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N], True for unvisited
    unvisited_mask = state.unvisited_mask()
    # [B], index of the current node
    current_node_idx = state.current_node_index()
    # [B, 1, 1] for broadcasting
    current_node_idx_b = current_node_idx.unsqueeze(-1).unsqueeze(-1)
    # [B, 1, N]
    current_node_idx_b_expanded = current_node_idx_b.expand(-1, 1, dist_matrix.shape[2])

    # --- Term 1: Expected distance to the next node ---
    # [B, N], distances from current node to all other nodes
    dists_from_current = torch.gather(dist_matrix, 1, current_node_idx_b_expanded).squeeze(1)
    # Apply mask to consider only unvisited nodes
    dists_from_current_to_unvisited = dists_from_current.masked_fill(~unvisited_mask, 0.0)
    # [B], number of unvisited nodes
    num_unvisited = unvisited_mask.float().sum(dim=1)
    # Avoid division by zero for terminal states
    num_unvisited_safe = torch.max(num_unvisited, torch.ones_like(num_unvisited))
    # [B], average distance from current to unvisited nodes
    avg_dist_to_next = dists_from_current_to_unvisited.sum(dim=1) / num_unvisited_safe
    # Set to 0 if no nodes are unvisited (terminal state)
    avg_dist_to_next = avg_dist_to_next.masked_fill(num_unvisited == 0, 0.0)

    # --- Term 2: Expected length of the remaining tour among unvisited nodes ---
    # [B, N, N] -> [B, N, N], mask out distances involving visited nodes
    unvisited_mask_2d = unvisited_mask.unsqueeze(2) & unvisited_mask.unsqueeze(1)
    unvisited_dists = dist_matrix.masked_fill(~unvisited_mask_2d, 0.0)
    # [B], sum of all distances between unvisited nodes
    sum_unvisited_dists = unvisited_dists.sum(dim=(1, 2))
    # [B], number of pairs of unvisited nodes
    num_pairs = num_unvisited * (num_unvisited - 1.0).clamp(min=0.0)
    num_pairs_safe = torch.max(num_pairs, torch.ones_like(num_pairs))
    # [B], average distance between any two unvisited nodes
    avg_inter_unvisited_dist = sum_unvisited_dists / num_pairs_safe
    # Scale by the number of remaining edges to form the rest of the tour
    # A simple heuristic: (N_unvisited - 1) edges needed.
    remaining_tour_len_est = avg_inter_unvisited_dist * (num_unvisited - 1.0).clamp(min=0.0)

    # --- Combine terms ---
    # The total estimated future cost is the sum of the two components.
    # The negative sign is because the environment uses negative rewards (costs).
    value = - (avg_dist_to_next + remaining_tour_len_est)

    return value.unsqueeze(-1)