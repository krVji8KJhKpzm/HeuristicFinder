# score=0.003070
# gamma=-0.100000
# code_hash=a610a2e6373c687890e46243aea5f2e1110d3d54cda996b0f647293c35dfc5b9
# stats: mse=8.9082; rmse=2.98466; mse_tsp100=325.755; mse_tsp20=8.9082; mse_tsp50=51.9263; mse_worst=325.755; rmse_tsp100=18.0487; rmse_tsp20=2.98466; rmse_tsp50=7.20599; rmse_worst=18.0487
# ALGORITHM: {Estimate future tour length by summing the minimum cost to connect each unvisited node to the convex hull of the current tour and adding the cost to return to the start from the current node.}
# THOUGHT: {Estimate future tour length by summing the minimum cost to connect each unvisited node to the convex hull of the current tour and adding the cost to return to the start from the current node.}
def phi(state):
    """
    Estimates the future tour length by combining geometric and graph-based heuristics.
    The value is composed of three main parts:
    1. For each unvisited node, find the minimum distance to the convex hull of the already visited nodes.
       This estimates the cost of inserting each remaining node into the existing tour.
    2. Sum these minimum insertion costs over all unvisited nodes.
    3. Add the direct distance from the current node back to the starting node to account for closing the tour.
    This approach models the expansion of the current tour to include all remaining nodes.

    Args:
        state (TSPStateView): The current state of the TSP environment.
    Returns:
        torch.Tensor: A scalar tensor [B, 1] representing the estimated future cost.
    """
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N, 2]
    coords = state.all_node_coords()
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    # [B, N]
    visited_mask = ~unvisited_mask
    B, N, _ = coords.shape
    device = coords.device

    # Handle terminal states where no nodes are unvisited
    is_done = torch.all(~unvisited_mask, dim=1)

    # 1. For each unvisited node, find its minimum connection cost to the set of visited nodes.
    # We can represent the connection cost as the minimum distance from an unvisited node
    # to any of the already visited nodes.

    # [B, N, N] -> [B, N, N]
    # Create a distance matrix where rows are unvisited nodes and columns are visited nodes.
    # Distances to unvisited columns are set to infinity.
    masked_dist = dist_matrix.clone()
    # [B, 1, N]
    visited_mask_exp = visited_mask.unsqueeze(1)
    # Set distances to non-visited nodes to infinity, so min finds the closest visited node.
    masked_dist.masked_fill_(~visited_mask_exp.expand_as(dist_matrix), torch.finfo(dist_matrix.dtype).max)

    # [B, N]
    # For each node (row index), find the minimum distance to any visited node (column index).
    min_dist_to_visited, _ = torch.min(masked_dist, dim=2)

    # 2. Sum these minimum connection costs, but only for the nodes that are currently unvisited.
    # For nodes already visited, their future connection cost is zero.
    min_dist_to_visited.masked_fill_(visited_mask, 0)
    # [B]
    sum_of_min_dists = torch.sum(min_dist_to_visited, dim=1)

    # 3. Add the cost to return to the start from the current location.
    # This is the cost of closing the final loop.
    # [B]
    current_node_idx = state.current_node_index()
    # [B]
    start_node_idx = state.first_node_index()
    # [B]
    dist_to_start = dist_matrix[torch.arange(B, device=device), current_node_idx, start_node_idx]

    # Total estimated future cost
    # [B]
    value = sum_of_min_dists + dist_to_start

    # If the tour is done, the future cost is zero.
    value.masked_fill_(is_done, 0.0)

    return value.unsqueeze(-1)