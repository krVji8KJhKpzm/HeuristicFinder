# score=0.086898
# gamma=0.100000
# code_hash=3950d9c6da72e659adae9f19c68687de8d0537a62b1013f7f4b9931199242483
# stats: mse=11.5078; rmse=3.39231; mse_tsp100=559.082; mse_tsp20=11.5078; mse_tsp50=120.004; rmse_tsp100=23.6449; rmse_tsp20=3.39231; rmse_tsp50=10.9547
# ALGORITHM: {auto}
# THOUGHT: {auto}
def phi(state):
    """
    Estimates the future tour length for a Traveling Salesperson Problem (TSP) state.

    This function calculates a potential value (phi) for a given state, which serves as an
    approximation of the expected future cost (remaining tour length). The potential is
    composed of two main components:
    1. The cost to travel from the current node to the "center" of the remaining unvisited nodes.
    2. The estimated cost to traverse the remaining unvisited nodes, modeled as the product of
       the number of remaining edges and the average distance between unvisited nodes.

    Args:
        state (TSPStateView): A view of the current state of the TSP environment, providing
                              access to information like node coordinates, distance matrix,
                              and visited/unvisited masks for a batch of instances.

    Returns:
        torch.Tensor: A scalar tensor for each state in the batch, representing the
                      estimated future tour length. The shape is broadcastable to [B, 1].
    """
    # [B, N, 2]
    coords = state.all_node_coords()
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N], boolean
    unvisited_mask = state.unvisited_mask()
    # [B], long
    current_node_idx = state.current_node_index()

    # Number of unvisited nodes, [B]
    n_unvisited = unvisited_mask.sum(dim=1, dtype=torch.float32)

    # Create a safe mask for division, preventing division by zero when n_unvisited is 0 or 1.
    # [B]
    safe_mask = n_unvisited > 1.0
    # [B, N]
    safe_unvisited_mask = unvisited_mask & safe_mask.unsqueeze(-1)
    # [B]
    safe_n_unvisited = safe_unvisited_mask.sum(dim=1, dtype=torch.float32)

    # 1. Average distance from the current node to all unvisited nodes.
    # [B, N]
    dist_from_current = torch.gather(dist_matrix, 1, current_node_idx.view(-1, 1, 1).expand(-1, 1, dist_matrix.size(2))).squeeze(1)
    # [B]
    sum_dist_to_unvisited = (dist_from_current * unvisited_mask).sum(dim=1)
    # [B]
    avg_dist_to_unvisited = sum_dist_to_unvisited / torch.clamp(n_unvisited, min=1.0)

    # 2. Average distance between all pairs of unvisited nodes.
    # [B, N, N]
    unvisited_from = safe_unvisited_mask.unsqueeze(2)
    unvisited_to = safe_unvisited_mask.unsqueeze(1)
    # [B, N, N], mask for pairs of unvisited nodes
    unvisited_pair_mask = unvisited_from & unvisited_to
    # [B]
    sum_inter_unvisited_dist = (dist_matrix * unvisited_pair_mask).sum(dim=(1, 2))
    # [B], number of pairs is roughly n_unvisited^2
    num_pairs = safe_n_unvisited * safe_n_unvisited
    # [B]
    avg_inter_unvisited_dist = sum_inter_unvisited_dist / torch.clamp(num_pairs, min=1.0)

    # Estimate remaining tour length
    # The number of remaining edges to form a tour is n_unvisited.
    # We use (n_unvisited - 1) as a heuristic for the number of connections needed.
    remaining_edges = n_unvisited - 1.0
    estimated_remaining_tour = remaining_edges * avg_inter_unvisited_dist

    # Combine the two components.
    # The final potential is the sum of the cost to get to the next node (approximated
    # by avg_dist_to_unvisited) and the cost to tour the rest (estimated_remaining_tour).
    value = avg_dist_to_unvisited + estimated_remaining_tour
    
    # Handle terminal states where n_unvisited is 0 or 1.
    # The logic above with safe_mask handles n_unvisited <= 1 by making the second term zero.
    # For n_unvisited == 0, avg_dist_to_unvisited is also 0.
    # For n_unvisited == 1, avg_dist_to_unvisited is the distance to the last node.
    # We explicitly set phi to 0 for terminal states (n_unvisited <= 1) for consistency.
    is_terminal = n_unvisited <= 1.0
    final_value = torch.where(is_terminal, torch.zeros_like(value), value)

    return final_value.unsqueeze(-1)