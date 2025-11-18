# score=3.406448
# gamma=-0.100000
# code_hash=a509be98760cefdd1fed21f2974aece62623fff8e8b93600894306f7a87ee209
# stats: mse=0.197561; rmse=0.444478; mse_tsp100=0.293561; mse_tsp20=0.197561; mse_tsp50=0.201841; mse_worst=0.293561; rmse_tsp100=0.541813; rmse_tsp20=0.444478; rmse_tsp50=0.449267; rmse_worst=0.541813
# ALGORITHM: {Estimate future tour length by summing the minimum spanning tree cost of unvisited nodes and adding the costs to connect the current and start nodes to this MST.}
# THOUGHT: {Estimate future tour length by summing the minimum spanning tree cost of unvisited nodes and adding the costs to connect the current and start nodes to this MST.}
def phi(state):
    """
    Estimates the future tour length for a TSP state.
    The estimation is based on three components:
    1. The cost of a Minimum Spanning Tree (MST) over all unvisited nodes.
    2. The minimum cost to connect the current node to one of the unvisited nodes.
    3. The minimum cost to connect the start node to one of the unvisited nodes.
    This heuristic approximates the cost of visiting all remaining nodes and returning home.
    Args:
        state: A view of the current state of the TSP environment.
    Returns:
        value: A tensor of shape [B, 1] representing the estimated future tour length.
    """
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    B, N, _ = dist_matrix.shape

    # 1. MST cost on unvisited nodes
    # Create a subgraph distance matrix for unvisited nodes
    # [B, N, N]
    unvisited_dist = dist_matrix.clone()
    # Mask rows and columns for visited nodes by setting distances to infinity
    unvisited_dist.masked_fill_(~unvisited_mask.unsqueeze(1), float('inf'))
    unvisited_dist.masked_fill_(~unvisited_mask.unsqueeze(2), float('inf'))

    # Prim's algorithm for MST cost
    # [B]
    num_unvisited = unvisited_mask.sum(dim=1)
    # [B, N], init with large values
    min_cost = torch.full_like(unvisited_mask, float('inf'), dtype=dist_matrix.dtype)
    # [B, N], init with False
    in_mst = torch.zeros_like(unvisited_mask, dtype=torch.bool)
    # [B]
    mst_cost = torch.zeros(B, device=dist_matrix.device)

    # Find the first unvisited node to start Prim's algorithm
    # [B], indices of first unvisited nodes
    start_node_idx = torch.where(unvisited_mask.any(1), unvisited_mask.float().argmax(1), -1)

    # Set the cost of the starting node to 0
    # Use scatter for batched indexing
    min_cost.scatter_(1, start_node_idx.unsqueeze(1).clamp(min=0), 0)

    # Prim's algorithm loop (vectorized)
    # This loop runs N times, which is more than necessary but avoids dynamic loops
    # and is safe because nodes already in MST won't be chosen again.
    for _ in range(N):
        # Select node `u` not in MST with the minimum cost
        # [B, N]
        cost_if_not_in_mst = min_cost.clone()
        cost_if_not_in_mst.masked_fill_(in_mst, float('inf'))
        # [B, 1]
        u_cost, u_idx = cost_if_not_in_mst.min(dim=1, keepdim=True)

        # Add cost to total and mark node as in MST
        # This mask handles batches where no unvisited nodes are left (u_cost is inf)
        is_finite_mask = torch.isfinite(u_cost.squeeze(-1))
        mst_cost[is_finite_mask] += u_cost[is_finite_mask].squeeze(-1)
        in_mst.scatter_(1, u_idx.clamp(min=0), True)

        # Update min_cost for neighbors of `u`
        # [B, 1, N]
        dist_from_u = torch.gather(unvisited_dist, 1, u_idx.unsqueeze(-1).expand(-1, 1, N))
        # [B, N]
        dist_from_u_squeezed = dist_from_u.squeeze(1)
        # [B, N]
        min_cost = torch.min(min_cost, dist_from_u_squeezed)

    # 2. Minimum cost from current node to any unvisited node
    # [B] -> [B, 1, 1] -> [B, 1, N]
    current_node_exp = state.current_node_index().view(B, 1, 1).expand(-1, 1, N)
    # [B, 1, N]
    dist_from_current = torch.gather(dist_matrix, 1, current_node_exp)
    # [B, N]
    dist_from_current_squeezed = dist_from_current.squeeze(1)
    dist_from_current_squeezed.masked_fill_(~unvisited_mask, float('inf'))
    # [B]
    min_to_unvisited = dist_from_current_squeezed.min(dim=1).values
    min_to_unvisited.masked_fill_(num_unvisited == 0, 0) # If no unvisited, cost is 0

    # 3. Minimum cost from start node to any unvisited node
    # [B] -> [B, 1, 1] -> [B, 1, N]
    first_node_exp = state.first_node_index().view(B, 1, 1).expand(-1, 1, N)
    # [B, 1, N]
    dist_from_start = torch.gather(dist_matrix, 1, first_node_exp)
    # [B, N]
    dist_from_start_squeezed = dist_from_start.squeeze(1)
    dist_from_start_squeezed.masked_fill_(~unvisited_mask, float('inf'))
    # [B]
    min_to_return = dist_from_start_squeezed.min(dim=1).values
    min_to_return.masked_fill_(num_unvisited == 0, 0) # If no unvisited, cost is 0

    # Total estimated cost
    # Special case: if only one unvisited node, MST cost is 0. The cost is just to go there and back.
    # [B]
    is_one_unvisited = (num_unvisited == 1)
    # [B, 1]
    first_unvisited_idx = torch.where(is_one_unvisited, unvisited_mask.float().argmax(1).unsqueeze(-1), -1)
    # [B, 1]
    dist_curr_to_last = torch.gather(dist_from_current_squeezed, 1, first_unvisited_idx.clamp(min=0))
    # [B, 1]
    dist_last_to_start = torch.gather(dist_from_start_squeezed, 1, first_unvisited_idx.clamp(min=0))
    # [B]
    one_unvisited_cost = (dist_curr_to_last + dist_last_to_start).squeeze(-1)

    # Combine costs
    value = torch.where(is_one_unvisited, one_unvisited_cost, mst_cost + min_to_unvisited + min_to_return)

    # Final state: tour complete, cost is 0
    value.masked_fill_(num_unvisited == 0, 0)

    return value.unsqueeze(-1)