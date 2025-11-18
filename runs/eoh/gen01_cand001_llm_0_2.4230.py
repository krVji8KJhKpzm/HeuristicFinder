# score=2.423034
# gamma=-0.100000
# code_hash=1b3b0812ca98a6aa6a8fb459b2e47dca3884b0f5419dcd0d7504221710b3f7b9
# stats: mse=0.239611; rmse=0.489501; mse_tsp100=0.412706; mse_tsp20=0.239611; mse_tsp50=0.249805; mse_worst=0.412706; rmse_tsp100=0.642422; rmse_tsp20=0.489501; rmse_tsp50=0.499805; rmse_worst=0.642422
# ALGORITHM: {Estimate the future tour length as the sum of distances from each unvisited node to its two nearest unvisited neighbors, plus connection costs from the current and start nodes.}
# THOUGHT: {Estimate the future tour length as the sum of distances from each unvisited node to its two nearest unvisited neighbors, plus connection costs from the current and start nodes.}
def phi(state):
    """
    Estimates future tour length by approximating a path through unvisited nodes.
    The value is composed of three parts:
    1. An approximation of the sub-tour length for unvisited nodes. For each unvisited node,
       we find the distance to its two nearest neighbors among the other unvisited nodes.
       The sum of these distances, averaged over all unvisited nodes, provides an estimate
       of the local path cost. This is scaled by the number of unvisited nodes.
    2. The minimum distance from the current node to any of the unvisited nodes.
    3. The minimum distance from the start node to any of the unvisited nodes.

    Args:
        state (TSPStateView): The current state of the TSP environment.
    Returns:
        torch.Tensor: A scalar tensor [B, 1] representing the estimated future cost.
    """
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    B, N, _ = dist_matrix.shape

    # Handle terminal and near-terminal states
    num_unvisited = unvisited_mask.sum(dim=1, keepdim=True)  # [B, 1]
    is_done = (num_unvisited == 0)
    is_last_step = (num_unvisited == 1)
    is_penultimate = (num_unvisited == 2)

    # 1. Approximate sub-tour length through unvisited nodes
    # Create a distance matrix only considering unvisited nodes.
    # We set distances to/from visited nodes to infinity.
    unvisited_dist = dist_matrix.clone()
    inf = torch.finfo(dist_matrix.dtype).max
    # Mask rows (from) and columns (to) corresponding to visited nodes
    unvisited_dist.masked_fill_(~unvisited_mask.unsqueeze(2), inf)
    unvisited_dist.masked_fill_(~unvisited_mask.unsqueeze(1), inf)

    # For each unvisited node, find the distances to its two nearest unvisited neighbors.
    # We use k=3 because the node itself is included with distance 0.
    # [B, N, 3]
    k = min(3, N) # handle N < 3 case
    topk_dists, _ = torch.topk(unvisited_dist, k, dim=2, largest=False)

    # The first value (k=0) is always 0 (distance to self), so we take k=1 and k=2.
    # Sum of distances to two nearest neighbors for each node.
    # If less than 3 unvisited nodes, some of these will be inf.
    if k > 2:
        two_nn_dist_sum = topk_dists[:, :, 1] + topk_dists[:, :, 2] # [B, N]
    elif k > 1:
        two_nn_dist_sum = topk_dists[:, :, 1] # [B, N]
    else:
        two_nn_dist_sum = torch.zeros_like(unvisited_mask, dtype=dist_matrix.dtype)


    # Mask out the sums for visited nodes
    two_nn_dist_sum.masked_fill_(~unvisited_mask, 0.0)

    # The sum of these distances over all unvisited nodes, divided by 2 (since each edge is counted twice),
    # approximates the MST length, which is a lower bound on the tour length.
    # We use the sum directly as a heuristic.
    subtour_cost = torch.sum(two_nn_dist_sum, dim=1) / 2.0

    # 2. Cost to connect the current node to the unvisited set
    # [B] -> [B, 1, 1] -> [B, 1, N]
    current_node_idx = state.current_node_index().view(B, 1, 1).expand(-1, 1, N)
    # [B, N]
    dist_from_current = torch.gather(dist_matrix, 1, current_node_idx).squeeze(1)
    # [B]
    min_dist_to_unvisited = torch.min(dist_from_current.masked_fill(~unvisited_mask, inf), dim=1).values

    # 3. Cost to connect the start node to the unvisited set
    # [B] -> [B, 1, 1] -> [B, 1, N]
    start_node_idx = state.first_node_index().view(B, 1, 1).expand(-1, 1, N)
    # [B, N]
    dist_from_start = torch.gather(dist_matrix, 1, start_node_idx).squeeze(1)
    # [B]
    min_dist_from_start = torch.min(dist_from_start.masked_fill(~unvisited_mask, inf), dim=1).values

    # Combine the components
    value = subtour_cost + min_dist_to_unvisited + min_dist_from_start

    # Handle edge cases where subtour_cost is not well-defined
    # If only two unvisited nodes, cost is current->A->B->start or current->B->A->start
    if torch.any(is_penultimate):
      # [B_pen, 2]
      unvisited_indices = unvisited_mask[is_penultimate.squeeze(-1)].long().nonzero(as_tuple=False)[:, 1].view(-1, 2)
      node_a_idx, node_b_idx = unvisited_indices[:, 0], unvisited_indices[:, 1]
      
      current_node_pen = state.current_node_index()[is_penultimate.squeeze(-1)]
      start_node_pen = state.first_node_index()[is_penultimate.squeeze(-1)]
      dist_matrix_pen = dist_matrix[is_penultimate.squeeze(-1)]

      dist_curr_a = dist_matrix_pen[torch.arange(len(current_node_pen)), current_node_pen, node_a_idx]
      dist_curr_b = dist_matrix_pen[torch.arange(len(current_node_pen)), current_node_pen, node_b_idx]
      dist_a_b = dist_matrix_pen[torch.arange(len(node_a_idx)), node_a_idx, node_b_idx]
      dist_a_start = dist_matrix_pen[torch.arange(len(start_node_pen)), node_a_idx, start_node_pen]
      dist_b_start = dist_matrix_pen[torch.arange(len(start_node_pen)), node_b_idx, start_node_pen]
      
      cost1 = dist_curr_a + dist_a_b + dist_b_start
      cost2 = dist_curr_b + dist_a_b + dist_a_start
      penultimate_cost = torch.min(cost1, cost2)
      value[is_penultimate.squeeze(-1)] = penultimate_cost

    # If only one unvisited node, cost is current -> last -> start
    if torch.any(is_last_step):
        # [B_last, 1]
        last_unvisited_idx = unvisited_mask[is_last_step.squeeze(-1)].long().argmax(dim=1, keepdim=True)
        # [B_last]
        dist_curr_to_last = torch.gather(dist_from_current[is_last_step.squeeze(-1)], 1, last_unvisited_idx).squeeze(-1)
        dist_last_to_start = torch.gather(dist_from_start[is_last_step.squeeze(-1)], 1, last_unvisited_idx).squeeze(-1)
        last_step_cost = dist_curr_to_last + dist_last_to_start
        value[is_last_step.squeeze(-1)] = last_step_cost
    
    # If the tour is done, future cost is 0
    value.masked_fill_(is_done.squeeze(-1), 0.0)
    # Replace any remaining infs (e.g., from min over empty set) with 0
    value = torch.nan_to_num(value, posinf=0.0)

    return value.unsqueeze(-1)