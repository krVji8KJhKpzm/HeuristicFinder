# score=0.012843
# gamma=-0.100000
# code_hash=fb7bdab678a355c5b634ad50ffb00335f41a66d8901f0875375b428e57a8e79e
# stats: mse=77.8606; rmse=8.82386; mse_tsp100=1123.3; mse_tsp20=77.8606; mse_tsp50=340.597; rmse_tsp100=33.5156; rmse_tsp20=8.82386; rmse_tsp50=18.4553
# ALGORITHM: {auto} def phi(state): """ Estimates the future tour length (Monte Carlo value V(state)) for the TSP environment. {My algorithm calculates the expected cost-to-go by summing two components: (1) the average distance from the current node to all unvisited nodes, representing the next step's expected cost, and (2) the expected cost of the remaining sub-tour, approximated by the number of remaining edges multiplied by the average pairwise distance between all unvisited nodes.}
# THOUGHT: {auto}
def phi(state):
    """
    Estimates the future tour length (Monte Carlo value V(state)) for the TSP environment.
    {My algorithm calculates the expected cost-to-go by summing two components: (1) the average distance from the current node to all unvisited nodes, representing the next step's expected cost, and (2) the expected cost of the remaining sub-tour, approximated by the number of remaining edges multiplied by the average pairwise distance between all unvisited nodes.}
    Args:
        state (TSPStateView): The current state of the TSP environment.
    Returns:
        torch.Tensor: A scalar potential value for each state in the batch, shape [B, 1].
    """
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N]
    unvisited = state.unvisited_mask()
    # [B]
    current_node = state.current_node_index()
    # [B, 1, N]
    current_node_exp = current_node.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, dist_matrix.size(2))

    # Number of unvisited nodes, [B]
    num_unvisited = unvisited.sum(dim=1, dtype=torch.float32)

    # Avoid division by zero for terminal states (num_unvisited can be 0 or 1)
    # If num_unvisited is 0 or 1, the future path length is 0.
    is_not_terminal = (num_unvisited > 1.0)
    safe_num_unvisited = torch.where(is_not_terminal, num_unvisited, torch.ones_like(num_unvisited))
    safe_num_unvisited_pairs = torch.where(is_not_terminal, num_unvisited * (num_unvisited - 1) / 2.0, torch.ones_like(num_unvisited))

    # 1. Cost from current node to the next unvisited node (approximate)
    # [B, 1, N]
    dist_from_current = torch.gather(dist_matrix, 1, current_node_exp)
    # [B, N]
    dist_from_current_squeezed = dist_from_current.squeeze(1)
    # Set distances to visited nodes to 0 to exclude them from the sum
    dist_from_current_to_unvisited = dist_from_current_squeezed * unvisited
    # [B]
    sum_dist_to_unvisited = dist_from_current_to_unvisited.sum(dim=1)
    # [B], average distance to an unvisited node
    avg_dist_to_next = sum_dist_to_unvisited / safe_num_unvisited

    # 2. Cost of the remaining sub-tour among unvisited nodes (approximate)
    # Create a mask for pairs of unvisited nodes
    # [B, N, 1] * [B, 1, N] -> [B, N, N]
    unvisited_pair_mask = unvisited.unsqueeze(2) * unvisited.unsqueeze(1)
    # [B, N, N], distances between unvisited nodes
    sub_tour_dists = dist_matrix * unvisited_pair_mask
    # [B], sum of distances between all pairs of unvisited nodes (upper triangle)
    sum_pairwise_dist_unvisited = sub_tour_dists.sum(dim=[1, 2]) / 2.0
    # [B], average distance between any two unvisited nodes
    avg_pairwise_dist = sum_pairwise_dist_unvisited / safe_num_unvisited_pairs

    # The number of edges in the remaining tour is (num_unvisited - 1)
    # We add 1 to connect back to the start, but let's use num_unvisited as an approximation
    # for the number of "steps" or edges remaining.
    remaining_edges = num_unvisited - 1.0
    
    # Approximate cost of the remaining sub-tour
    # Using num_unvisited instead of remaining_edges as it seems more stable
    remaining_tour_cost = num_unvisited * avg_pairwise_dist

    # Total estimated future cost
    # We negate because the environment uses negative distances as rewards.
    # Phi should be an estimate of V, which is the sum of future rewards (negative lengths).
    # So, V is negative. We return a negative value.
    value = -(avg_dist_to_next + remaining_tour_cost)

    # For terminal states, future cost is 0.
    final_value = torch.where(is_not_terminal, value, torch.zeros_like(value))

    return final_value.unsqueeze(-1)