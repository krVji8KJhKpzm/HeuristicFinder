# score=2.166641
# gamma=1.000000
# code_hash=215a9bb222ba5cb837cab1ca242841eb364c178adfc78eeff4dc565d3e0b0870
# stats: mse=0.461544; rmse=0.67937; mse_tsp100=2.07279; mse_tsp20=0.461544; mse_tsp50=1.10612; rmse_tsp100=1.43972; rmse_tsp20=0.67937; rmse_tsp50=1.05172
# ALGORITHM: {auto}
# THOUGHT: {auto}
def phi(state):
    """
    Estimates the future tour length (cost-to-go) for a TSP state.

    The potential is calculated as the sum of two components:
    1. The cost to travel from the current node to the nearest unvisited node.
    2. An estimate of the cost to complete the tour through the remaining unvisited nodes.
       This is approximated by summing, for each unvisited node, the minimum distance
       to any *other* unvisited node. This serves as a rough lower bound on the
       remaining tour length, similar to an MST-based heuristic but simpler to compute
       and differentiable.

    Args:
        state: A TSPStateView object providing access to the environment state.

    Returns:
        A tensor of shape [B, 1] representing the estimated future cost for each state in the batch.
    """
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    # [B]
    current_node = state.current_node_index()

    # Create a mask for valid destinations: must be unvisited.
    # [B, N] -> [B, 1, N]
    unvisited_mask_from = unvisited_mask.unsqueeze(1)
    # [B, N] -> [B, N, 1]
    unvisited_mask_to = unvisited_mask.unsqueeze(2)

    # Component 1: Distance from the current node to the nearest unvisited node.
    # Get distances from the current node: [B, N]
    current_dists = torch.gather(dist_matrix, 1, current_node.view(-1, 1, 1).expand(-1, 1, dist_matrix.size(2))).squeeze(1)
    # Apply mask to consider only unvisited destinations
    current_dists_masked = torch.where(unvisited_mask, current_dists, torch.full_like(current_dists, float('inf')))
    # Find the minimum distance to an unvisited node.
    # If no nodes are unvisited (terminal state), this will be inf, which we handle.
    min_dist_to_unvisited = torch.min(current_dists_masked, dim=1, keepdim=True).values

    # Component 2: Estimated cost for the remaining tour of unvisited nodes.
    # Create a distance matrix containing only edges between unvisited nodes.
    # [B, N, N]
    unvisited_dist_matrix = dist_matrix.where(unvisited_mask_from & unvisited_mask_to, torch.full_like(dist_matrix, float('inf')))
    # To prevent a node from connecting to itself, set the diagonal to infinity.
    # This is crucial for finding the minimum distance to *another* unvisited node.
    # This is redundant since dist_matrix diagonal is 0, but good for clarity.
    unvisited_dist_matrix.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))

    # For each unvisited node (row), find the minimum distance to any other unvisited node (column).
    # [B, N]
    min_dists_from_unvisited, _ = torch.min(unvisited_dist_matrix, dim=2)
    # Zero out the values for nodes that have already been visited.
    min_dists_from_unvisited = torch.where(unvisited_mask, min_dists_from_unvisited, 0.0)
    # Sum these minimum distances to get an estimate of the remaining tour cost.
    # [B, 1]
    remaining_tour_cost_estimate = min_dists_from_unvisited.sum(dim=1, keepdim=True)

    # Total potential is the sum of the two components.
    # For terminal states, unvisited_mask is all False.
    # min_dist_to_unvisited will be inf, and remaining_tour_cost_estimate will be 0.
    # We want phi to be 0 at terminal states.
    value = min_dist_to_unvisited + remaining_tour_cost_estimate

    # Handle terminal states: if there are no unvisited nodes, the potential should be 0.
    # `torch.any(unvisited_mask, dim=1)` is True if there's at least one unvisited node.
    is_not_terminal = torch.any(unvisited_mask, dim=1, keepdim=True)
    value = torch.where(is_not_terminal, value, 0.0)

    return value