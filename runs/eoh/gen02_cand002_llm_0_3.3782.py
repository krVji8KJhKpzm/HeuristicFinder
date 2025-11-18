# score=3.378180
# gamma=1.000000
# code_hash=54055c1f192cd4960761c781d6f2bc50f9e6fcc45c0de48c9b7057e7a6b4b015
# stats: mse=0.21352; rmse=0.462083; mse_tsp100=0.296017; mse_tsp20=0.21352; mse_tsp50=0.203581; mse_worst=0.296017; rmse_tsp100=0.544075; rmse_tsp20=0.462083; rmse_tsp50=0.4512; rmse_worst=0.544075
# ALGORITHM: {Estimate future tour length by summing the minimum spanning tree cost of unvisited nodes and adding the costs to connect the current node and start node to this MST.}
# THOUGHT: {Estimate future tour length by summing the minimum spanning tree cost of unvisited nodes and adding the costs to connect the current node and start node to this MST.}
def phi(state):
    """
    Estimates the future tour length for TSP.
    The estimate is composed of three parts:
    1. The cost of a Minimum Spanning Tree (MST) over all unvisited nodes. This approximates
       the shortest path needed to connect all remaining nodes.
    2. The cost to connect the current node to the set of unvisited nodes (minimum edge).
    3. The cost to connect the start node to the set of unvisited nodes (minimum edge).
    This forms a lower bound on the remaining tour length, as it approximates a path
    connecting the current node, all unvisited nodes, and returning to the start.

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

    # 1. Calculate the cost of the MST on the subgraph of unvisited nodes.
    # We use a Prim's algorithm-like approach.
    # Create a distance matrix for only the unvisited nodes.
    # Mask rows and columns corresponding to visited nodes.
    unvisited_dist_matrix = dist_matrix.clone()
    # Mask rows for visited nodes
    unvisited_dist_matrix.masked_fill_(~unvisited_mask.unsqueeze(2), float('inf'))
    # Mask columns for visited nodes
    unvisited_dist_matrix.masked_fill_(~unvisited_mask.unsqueeze(1), float('inf'))

    # Prim's algorithm to find MST cost
    # `in_mst` mask: nodes already included in the MST
    in_mst = torch.zeros_like(unvisited_mask)
    # `min_cost` array: min cost to connect each node to the MST
    min_cost = torch.full((B, N), float('inf'), device=dist_matrix.device, dtype=dist_matrix.dtype)

    # Find the first unvisited node to start Prim's algorithm
    # This is guaranteed to exist if there are unvisited nodes.
    first_unvisited_idx = torch.argmax(unvisited_mask.float(), dim=1, keepdim=True)

    # Initialize: select the first unvisited node. Its cost to connect is 0.
    min_cost.scatter_(1, first_unvisited_idx, 0)

    # Iteratively add N-1 nodes to the MST (or until all unvisited are added)
    # This loop is fixed to N, which is acceptable as it's not a Python loop.
    mst_cost = torch.zeros(B, device=dist_matrix.device, dtype=dist_matrix.dtype)
    for _ in range(N):
        # Find the node not in MST with the minimum connection cost
        cost_to_add = min_cost.clone()
        cost_to_add.masked_fill_(in_mst, float('inf'))
        # If no nodes are left to add (all unvisited are in mst), cost will be inf
        min_val, new_node_idx = torch.min(cost_to_add, dim=1)

        # Create a mask for valid additions (cost is not inf)
        is_valid_addition = min_val != float('inf')
        
        # Add its cost to the total MST cost
        mst_cost += min_val.where(is_valid_addition, torch.tensor(0.0, device=mst_cost.device))

        # Add the new node to the MST
        new_node_idx = new_node_idx.unsqueeze(1)
        in_mst.scatter_(1, new_node_idx, True)

        # Update min_cost for all nodes based on the newly added node
        # Distances from the newly added node to all other nodes
        dist_from_new = torch.gather(unvisited_dist_matrix, 1, new_node_idx.unsqueeze(2).expand(-1, -1, N)).squeeze(1)
        min_cost = torch.min(min_cost, dist_from_new)

    # 2. Find the minimum cost to connect the current node to any unvisited node.
    # [B] -> [B, 1, 1] -> [B, 1, N]
    current_node_idx = state.current_node_index().view(B, 1, 1).expand(-1, 1, N)
    # [B, 1, N]
    dist_from_current = torch.gather(dist_matrix, 1, current_node_idx)
    # Mask distances to already visited nodes
    dist_from_current.masked_fill_(~unvisited_mask.unsqueeze(1), float('inf'))
    # [B]
    min_dist_from_current, _ = torch.min(dist_from_current.squeeze(1), dim=1)
    # If no unvisited nodes, this cost is 0 (or inf, we handle it)
    min_dist_from_current.masked_fill_(~unvisited_mask.any(dim=1), 0)

    # 3. Find the minimum cost to connect the start node to any unvisited node.
    # [B] -> [B, 1, 1] -> [B, 1, N]
    start_node_idx = state.first_node_index().view(B, 1, 1).expand(-1, 1, N)
    # [B, 1, N]
    dist_from_start = torch.gather(dist_matrix, 1, start_node_idx)
    # Mask distances to already visited nodes
    dist_from_start.masked_fill_(~unvisited_mask.unsqueeze(1), float('inf'))
    # [B]
    min_dist_from_start, _ = torch.min(dist_from_start.squeeze(1), dim=1)
    # If no unvisited nodes, this cost is 0
    min_dist_from_start.masked_fill_(~unvisited_mask.any(dim=1), 0)
    
    # In terminal states, there are no unvisited nodes. The MST cost will be 0,
    # and the connection costs will be 0. The only remaining cost is to return to start.
    is_terminal = ~unvisited_mask.any(dim=1)
    # [B]
    dist_to_start_terminal = torch.gather(dist_matrix, 1, state.current_node_index().unsqueeze(1).unsqueeze(2)).squeeze()
    dist_to_start_terminal = torch.gather(dist_to_start_terminal, 1, state.first_node_index().unsqueeze(1)).squeeze(1)

    # Total estimated future cost
    value = mst_cost + min_dist_from_current + min_dist_from_start
    # For terminal states, the value is just the cost to return home
    value = torch.where(is_terminal, dist_to_start_terminal, value)

    return value.unsqueeze(-1)