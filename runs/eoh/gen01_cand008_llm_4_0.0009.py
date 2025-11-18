# score=0.000888
# gamma=1.000000
# code_hash=ed09708794992a726ec22800dace99f18abcd13ae79deddbf39361787530f72b
# stats: mse=78.2019; rmse=8.84318; mse_tsp100=1125.78; mse_tsp20=78.2019; mse_tsp50=341.763; mse_worst=1125.78; rmse_tsp100=33.5527; rmse_tsp20=8.84318; rmse_tsp50=18.4868; rmse_worst=33.5527
# ALGORITHM: {auto}
# THOUGHT: {auto}
def phi(state):
    """
    Estimates the future tour length (value) for a TSP state.

    Args:
        state: A TSPStateView object with batch-friendly helper methods.

    Returns:
        A torch tensor of shape [B, 1] representing the estimated future cost.
    """
    # [B, N, 2]
    coords = state.all_node_coords()
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N], True for unvisited
    unvisited_mask = state.unvisited_mask()
    # [B], index of current node
    current_node = state.current_node_index()
    # [B], index of first node
    start_node = state.first_node_index()

    # Number of unvisited nodes, [B]
    num_unvisited = unvisited_mask.sum(dim=1, dtype=torch.float32)

    # Mask to prevent division by zero for terminal states
    is_not_terminal = num_unvisited > 0
    # Create a safe divisor, [B]
    safe_num_unvisited = torch.where(is_not_terminal, num_unvisited, torch.ones_like(num_unvisited))

    # --- Component 1: Cost from current node to the "center" of unvisited nodes ---
    # [B, 1, N]
    current_node_expanded = current_node.unsqueeze(1).unsqueeze(2).expand(-1, 1, dist_matrix.size(2))
    # [B, N], distances from current node to all other nodes
    dist_from_current = dist_matrix.gather(1, current_node_expanded).squeeze(1)
    # Mask out distances to already visited nodes
    dist_from_current_to_unvisited = dist_from_current * unvisited_mask
    # [B], average distance from current to unvisited
    avg_dist_to_unvisited = dist_from_current_to_unvisited.sum(dim=1) / safe_num_unvisited

    # --- Component 2: Estimated cost to traverse remaining unvisited nodes ---
    # Create a mask for pairs of unvisited nodes [B, N, N]
    unvisited_pairs_mask = unvisited_mask.unsqueeze(2) & unvisited_mask.unsqueeze(1)
    # Mask out diagonal and visited nodes
    unvisited_pairs_mask.diagonal(dim1=-2, dim2=-1).fill_(False)
    # [B, N, N], distances only between unvisited nodes
    unvisited_dist = dist_matrix * unvisited_pairs_mask
    # [B], sum of distances between all unique pairs of unvisited nodes
    sum_pairwise_unvisited_dist = unvisited_dist.sum(dim=(1, 2)) / 2.0
    # [B], number of pairs of unvisited nodes
    num_unvisited_pairs = num_unvisited * (num_unvisited - 1) / 2.0
    safe_num_unvisited_pairs = torch.where(num_unvisited_pairs > 0, num_unvisited_pairs, torch.ones_like(num_unvisited_pairs))
    # [B], average distance between unvisited nodes
    avg_pairwise_unvisited_dist = sum_pairwise_unvisited_dist / safe_num_unvisited_pairs
    # Scale by (num_unvisited - 1) to approximate a path through them
    internal_tour_cost = avg_pairwise_unvisited_dist * torch.clamp(num_unvisited - 1, min=0)

    # --- Component 3: Cost to return to the start from the "last" unvisited node ---
    # [B, 1, N]
    start_node_expanded = start_node.unsqueeze(1).unsqueeze(2).expand(-1, 1, dist_matrix.size(2))
    # [B, N], distances from start node to all other nodes
    dist_from_start = dist_matrix.gather(1, start_node_expanded).squeeze(1)
    # Mask out distances to already visited nodes
    dist_from_start_to_unvisited = dist_from_start * unvisited_mask
    # [B], average distance from start to unvisited
    avg_dist_from_start = dist_from_start_to_unvisited.sum(dim=1) / safe_num_unvisited

    # Combine components
    # The total estimated cost is the sum of the three parts.
    # We use a negative sign because value is typically anti-correlated with cost.
    # The logic is: travel to the unvisited cluster, travel within it, then return home.
    total_cost = avg_dist_to_unvisited + internal_tour_cost + avg_dist_from_start

    # For terminal states (num_unvisited == 0), the future cost is 0.
    # For the penultimate state (num_unvisited == 1), the cost is dist(current, last_node) + dist(last_node, start).
    # Let's check the penultimate case: num_unvisited=1.
    # avg_dist_to_unvisited = dist(current, last_node)
    # internal_tour_cost = 0
    # avg_dist_from_start = dist(start, last_node)
    # This seems reasonable.
    final_value = -total_cost * is_not_terminal

    return final_value.unsqueeze(-1)