# score=3.420970
# gamma=-0.100000
# code_hash=d5014fe6247751116f6de4174accdec9f7c2eafdcfcbe2f724c5ad25941cb364
# stats: mse=0.186826; rmse=0.432233; mse_tsp100=0.292315; mse_tsp20=0.186826; mse_tsp50=0.195723; mse_worst=0.292315; rmse_tsp100=0.540661; rmse_tsp20=0.432233; rmse_tsp50=0.442406; rmse_worst=0.540661
# ALGORITHM: {Estimate the future tour length by summing the MST cost of the unvisited nodes, the cost to connect the current node to this MST, and the cost to connect the start node to this MST for the return trip.}
# THOUGHT: {Estimate the future tour length by summing the MST cost of the unvisited nodes, the cost to connect the current node to this MST, and the cost to connect the start node to this MST for the return trip.}
def phi(state):
    """
    Estimates the future tour length using a Minimum Spanning Tree (MST) based heuristic.
    The cost is the sum of three components:
    1. The cost of the MST of the subgraph formed by the unvisited nodes. This provides a
       tight lower bound on the optimal sub-tour length for these nodes.
    2. The minimum cost to connect the current node to the set of unvisited nodes.
    3. The minimum cost to connect the start node to the set of unvisited nodes,
       which is necessary for closing the tour.
    
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
    # [B, N, 2]
    coords = state.all_node_coords()
    B, N, _ = coords.shape
    device = coords.device

    # Number of unvisited nodes, [B]
    n_unvisited = unvisited_mask.sum(dim=1)
    is_terminal = (n_unvisited == 0)
    
    # 1. Calculate the cost of the MST on the unvisited nodes.
    # Prim's algorithm for dense graphs.
    # We will compute this only for batches where there are unvisited nodes.
    # [B, N]
    min_cost = torch.full((B, N), float('inf'), device=device)
    # [B, N]
    visited_in_mst = torch.zeros_like(unvisited_mask)
    # [B]
    mst_cost = torch.zeros(B, device=device)
    
    # Initialize Prim's: pick the first unvisited node as the starting point for the MST.
    # Find the index of the first 'True' in unvisited_mask for each batch item.
    first_unvisited_idx = torch.argmax(unvisited_mask.int(), dim=1)
    
    # Set the min_cost for the starting node of the MST to 0.
    # This ensures it's the first one picked.
    min_cost.scatter_(1, first_unvisited_idx.unsqueeze(1), 0)
    
    # Mask min_cost for nodes that are already visited in the main tour.
    min_cost.masked_fill_(~unvisited_mask, float('inf'))

    # Prim's algorithm loop executed in a parallelized way over N-1 steps.
    # This is a fixed-iteration loop, avoiding dynamic Python loops.
    for _ in range(N):
        # Find the node `u` with the minimum cost that is unvisited in the MST.
        # [B, 1]
        u_idx = torch.argmin(min_cost, dim=1, keepdim=True)
        # [B]
        u_cost = torch.gather(min_cost, 1, u_idx).squeeze(1)
        
        # Add its cost to the total MST cost, handling inf for disconnected components.
        mst_cost += torch.where(u_cost == float('inf'), 0, u_cost)
        
        # Mark `u` as visited in the MST and remove it from consideration.
        visited_in_mst.scatter_(1, u_idx, True)
        min_cost.scatter_(1, u_idx, float('inf'))

        # Update min_costs for neighbors of `u`.
        # [B, N]
        dist_from_u = torch.gather(dist_matrix, 1, u_idx.unsqueeze(2).expand(-1, -1, N)).squeeze(1)
        
        # Update only if the new path through `u` is shorter and the neighbor is unvisited in MST.
        # We also need to ensure the neighbor is part of the unvisited set for the main tour.
        update_mask = (dist_from_u < min_cost) & ~visited_in_mst & unvisited_mask
        min_cost[update_mask] = dist_from_u[update_mask]

    # 2. Cost to connect the current node to the unvisited set.
    # [B, N]
    dist_from_current = torch.gather(dist_matrix, 1, current_node.view(-1, 1, 1).expand(-1, 1, N)).squeeze(1)
    # Mask distances to already visited nodes.
    dist_from_current.masked_fill_(~unvisited_mask, float('inf'))
    # [B]
    min_dist_to_unvisited, _ = torch.min(dist_from_current, dim=1)

    # 3. Cost to connect the start node to the unvisited set (for closing the loop).
    # [B, N]
    dist_from_start = torch.gather(dist_matrix, 1, start_node.view(-1, 1, 1).expand(-1, 1, N)).squeeze(1)
    dist_from_start.masked_fill_(~unvisited_mask, float('inf'))
    # [B]
    min_dist_from_start, _ = torch.min(dist_from_start, dim=1)

    # Total estimated cost
    # For the case with only one unvisited node, MST cost is 0. The formula becomes
    # dist(current -> last) + dist(start -> last), which is correct.
    value = mst_cost + min_dist_to_unvisited + min_dist_from_start

    # For terminal states, future cost is 0.
    value.masked_fill_(is_terminal, 0.0)
    # Replace any remaining infs (e.g., from min over empty set) with 0.
    value = torch.nan_to_num(value, posinf=0.0)

    return value.unsqueeze(-1)