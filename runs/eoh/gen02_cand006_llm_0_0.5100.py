# score=0.509967
# gamma=0.100000
# code_hash=c6aa769dbe2000ef83f865ad1a9c3185a3f1e99084e3c90a1ff0a0dfe96f6796
# stats: mse=0.490892; rmse=0.700637; mse_tsp100=1.96091; mse_tsp20=0.490892; mse_tsp50=1.0417; mse_worst=1.96091; rmse_tsp100=1.40033; rmse_tsp20=0.700637; rmse_tsp50=1.02064; rmse_worst=1.40033
# ALGORITHM: {Estimate the future tour length as the sum of distances from each unvisited node to its nearest neighbor among the unvisited, current, and start nodes, plus the minimum cost to connect the current and start nodes into this set.}
# THOUGHT: {Estimate the future tour length as the sum of distances from each unvisited node to its nearest neighbor among the unvisited, current, and start nodes, plus the minimum cost to connect the current and start nodes into this set.}
def phi(state):
    """
    Estimates the future tour length based on a nearest-neighbor heuristic within the unvisited set,
    also considering connections to the current and start nodes.
    The cost is the sum of three components:
    1. For each unvisited node, find the minimum distance to any other unvisited node. Sum these minimum distances.
       This approximates the cost of connecting all unvisited nodes together.
    2. The minimum cost to connect the current node to the set of unvisited nodes.
    3. The minimum cost to connect the start node to the set of unvisited nodes.
    This differs from MST by using a simpler, more local connection cost approximation.
    
    Args:
        state: A TSPStateView object with batch-friendly helper methods.

    Returns:
        A torch tensor of shape [B, 1] representing the estimated future tour length.
    """
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    # [B]
    current_node = state.current_node_index()
    # [B]
    start_node = state.first_node_index()
    # [B, N, 2], used to get B and N
    coords = state.all_node_coords()
    B, N, _ = coords.shape

    n_unvisited = unvisited_mask.sum(dim=1)
    is_terminal = (n_unvisited == 0)

    # 1. Approximate the cost to connect all unvisited nodes.
    # For each unvisited node, find the distance to its nearest neighbor *that is also unvisited*.
    # [B, N, N]
    unvisited_dist = dist_matrix.clone()
    # Mask rows and columns corresponding to visited nodes.
    # To find min distance from unvisited `i` to unvisited `j`, we need row `i` and col `j` to be valid.
    unvisited_row_mask = unvisited_mask.unsqueeze(2).expand(-1, -1, N)
    unvisited_col_mask = unvisited_mask.unsqueeze(1).expand(-1, N, -1)
    # Set distances to/from visited nodes to infinity.
    unvisited_dist.masked_fill_(~unvisited_row_mask, float('inf'))
    unvisited_dist.masked_fill_(~unvisited_col_mask, float('inf'))
    # Set diagonal to infinity to avoid picking the node itself.
    unvisited_dist.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))
    
    # [B, N], min distance for each node to its nearest unvisited neighbor.
    min_dists_to_unvisited, _ = torch.min(unvisited_dist, dim=2)
    
    # Sum these minimum distances. This is our approximation of the sub-tour cost.
    # We only sum over the unvisited nodes.
    # Replace inf with 0 for visited nodes so they don't contribute to the sum.
    min_dists_to_unvisited.masked_fill_(~unvisited_mask, 0.0)
    # [B]
    subtour_cost_approx = min_dists_to_unvisited.sum(dim=1)

    # 2. Cost to connect the current node to the unvisited set.
    # [B, N]
    dist_from_current = torch.gather(dist_matrix, 1, current_node.view(-1, 1, 1).expand(-1, 1, N)).squeeze(1)
    dist_from_current.masked_fill_(~unvisited_mask, float('inf'))
    # [B]
    min_dist_to_unvisited, _ = torch.min(dist_from_current, dim=1)

    # 3. Cost to connect the start node to the unvisited set.
    # [B, N]
    dist_from_start = torch.gather(dist_matrix, 1, start_node.view(-1, 1, 1).expand(-1, 1, N)).squeeze(1)
    dist_from_start.masked_fill_(~unvisited_mask, float('inf'))
    # [B]
    min_dist_from_start, _ = torch.min(dist_from_start, dim=1)

    # Total estimated cost
    value = subtour_cost_approx + min_dist_to_unvisited + min_dist_from_start

    # Handle terminal states and potential infs
    value.masked_fill_(is_terminal, 0.0)
    value = torch.nan_to_num(value, posinf=0.0)

    return value.unsqueeze(-1)