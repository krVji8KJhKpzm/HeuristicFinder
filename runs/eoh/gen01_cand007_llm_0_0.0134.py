# score=0.013379
# gamma=0.100000
# code_hash=6a5852e5d396822bcba2058559cc394425eefcabcf728af2a303c083fd70961e
# stats: mse=21.7091; rmse=4.6593; mse_tsp100=74.7425; mse_tsp20=21.7091; mse_tsp50=41.8858; mse_worst=74.7425; rmse_tsp100=8.64537; rmse_tsp20=4.6593; rmse_tsp50=6.47192; rmse_worst=8.64537
# ALGORITHM: {Estimate the future tour length by summing, for each unvisited node, half the distance to its two nearest unvisited neighbors, plus the costs to connect the current and start nodes into this estimated sub-tour.}
# THOUGHT: {Estimate the future tour length by summing, for each unvisited node, half the distance to its two nearest unvisited neighbors, plus the costs to connect the current and start nodes into this estimated sub-tour.}
def phi(state):
    """
    Estimates the future tour length (cost-to-go) for the TSP using a nearest-neighbor heuristic on the unvisited subgraph.
    The value is composed of three parts:
    1. An estimation of the sub-tour length through unvisited nodes. For each unvisited node, we find its two nearest
       unvisited neighbors and add half the distance to each (approximating one incoming and one outgoing edge).
    2. The minimum distance from the current node to any unvisited node (cost to enter the sub-tour).
    3. The minimum distance from the start node to any unvisited node (cost to return from the sub-tour).

    Args:
        state (TSPStateView): The current state of the TSP environment.

    Returns:
        torch.Tensor: A scalar tensor [B, 1] representing the estimated future cost.
    """
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    # [B, 1]
    num_unvisited = unvisited_mask.sum(dim=1, keepdim=True)

    # 1. Estimate the sub-tour length through unvisited nodes
    # Create a distance matrix masked to only show distances between unvisited nodes
    # [B, N, 1]
    unvisited_from = unvisited_mask.unsqueeze(-1)
    # [B, 1, N]
    unvisited_to = unvisited_mask.unsqueeze(-2)
    # [B, N, N]
    unvisited_dist = dist_matrix.masked_fill(~(unvisited_from & unvisited_to), float('inf'))
    # Set diagonal to infinity to avoid picking a node as its own neighbor
    unvisited_dist.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))

    # Find the two nearest unvisited neighbors for each unvisited node
    # [B, N, 2]
    k_smallest_dists, _ = torch.topk(unvisited_dist, k=2, dim=-1, largest=False)

    # Sum of distances to the two nearest unvisited neighbors, for unvisited nodes only
    # [B, N]
    sum_2_nn_dists = k_smallest_dists.sum(dim=-1)
    # Zero out distances for nodes that are already visited
    sum_2_nn_dists.masked_fill_(~unvisited_mask, 0.0)
    # We take the sum over all nodes and divide by 2, as each edge (i,j) is counted twice (once for i, once for j).
    # This is equivalent to summing (d1+d2)/2 for each node.
    # [B, 1]
    subtour_estimate = sum_2_nn_dists.sum(dim=-1, keepdim=True) * 0.5
    # Handle cases with < 2 unvisited nodes where topk(2) is not meaningful
    subtour_estimate = subtour_estimate.masked_fill(num_unvisited < 2, 0.0)

    # 2. & 3. Calculate connection costs from current and start nodes to the unvisited set
    # [B] -> [B, 1]
    current_node_idx = state.current_node_index().unsqueeze(-1)
    start_node_idx = state.first_node_index().unsqueeze(-1)
    # [B, N]
    dist_from_current = torch.gather(dist_matrix, 1, current_node_idx.unsqueeze(-1).expand(-1, -1, dist_matrix.size(-1))).squeeze(1)
    dist_from_start = torch.gather(dist_matrix, 1, start_node_idx.unsqueeze(-1).expand(-1, -1, dist_matrix.size(-1))).squeeze(1)

    # Find minimum distance to any unvisited node
    # [B, 1]
    min_dist_to_unvisited = torch.min(dist_from_current.masked_fill(~unvisited_mask, float('inf')), dim=1, keepdim=True).values
    min_dist_from_unvisited_to_start = torch.min(dist_from_start.masked_fill(~unvisited_mask, float('inf')), dim=1, keepdim=True).values
    
    # Handle terminal states where no unvisited nodes exist
    min_dist_to_unvisited = min_dist_to_unvisited.nan_to_num(posinf=0.0)
    min_dist_from_unvisited_to_start = min_dist_from_unvisited_to_start.nan_to_num(posinf=0.0)

    # Total estimated future cost
    value = subtour_estimate + min_dist_to_unvisited + min_dist_from_unvisited_to_start

    # If the tour is done (num_unvisited == 0), future cost is zero.
    # This is handled naturally by the components becoming 0.
    # If only one node is left, subtour_estimate is 0, and the cost is curr->last + last->start.
    # This is also handled naturally.

    return -value