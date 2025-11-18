# score=2.527233
# gamma=0.100000
# code_hash=dc604ef673fe9a7e63912216f576b19094fd380b26a2e04c3e2c3d74e5ee3b99
# stats: mse=0.39569; rmse=0.629039; mse_tsp100=0.179929; mse_tsp20=0.39569; mse_tsp50=0.201461; mse_worst=0.39569; rmse_tsp100=0.424181; rmse_tsp20=0.629039; rmse_tsp50=0.448845; rmse_worst=0.629039
# ALGORITHM: {Estimate the future tour length by summing the average distance from each unvisited node to its k-nearest unvisited neighbors, plus the cost to connect the current and start nodes to this unvisited cluster.}
# THOUGHT: {Estimate the future tour length by summing the average distance from each unvisited node to its k-nearest unvisited neighbors, plus the cost to connect the current and start nodes to this unvisited cluster.}
def phi(state):
    """
    Estimates future tour length based on local connectivity of unvisited nodes.
    The value is the sum of three components:
    1. For each unvisited node, calculate the average distance to its k-nearest
       neighbors that are also unvisited. Summing these averages provides a
       proxy for the sub-tour cost, similar to but simpler than MST.
    2. The minimum cost to connect the current node to any unvisited node.
    3. The minimum cost to connect the start node to any unvisited node,
       approximating the cost of closing the tour.
    This uses all_node_coords() to get the distance matrix.

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

    # Handle terminal states where no nodes are unvisited
    n_unvisited = unvisited_mask.sum(dim=1)
    is_terminal = (n_unvisited == 0)

    # 1. Estimate sub-tour cost for unvisited nodes using k-nearest neighbors.
    # [B, N, N]
    masked_dist = dist_matrix.clone()
    # Mask connections to/from already visited nodes.
    # [B, 1, N]
    unvisited_mask_from = unvisited_mask.unsqueeze(1)
    # [B, N, 1]
    unvisited_mask_to = unvisited_mask.unsqueeze(2)
    # Set distances to/from visited nodes to infinity.
    masked_dist.masked_fill_(~unvisited_mask_from | ~unvisited_mask_to, float('inf'))
    # Set diagonal to infinity to exclude self-loops.
    masked_dist.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))

    # Sort distances to find k-nearest neighbors for each unvisited node.
    # [B, N, N]
    sorted_dist, _ = torch.sort(masked_dist, dim=2)

    # Use a dynamic k, e.g., min(3, number of other unvisited nodes)
    # This avoids issues when few nodes are left.
    k = torch.min(torch.tensor(3, device=device), n_unvisited.clamp(min=1) - 1)
    k = k.clamp(min=1) # Ensure k is at least 1 for non-terminal states.

    # [B, N, k_max=3]
    k_nearest_dists = sorted_dist[:, :, :3]

    # Create a mask for averaging based on dynamic k.
    # [B, 1, 3]
    k_range = torch.arange(3, device=device).view(1, 1, 3)
    # [B, N, 3]
    k_mask = k_range < k.view(B, 1, 1)

    # Calculate average distance to k-nearest unvisited neighbors.
    # [B, N, 3]
    k_nearest_dists.masked_fill_(~k_mask, 0.0)
    # [B, N]
    sum_k_dists = k_nearest_dists.sum(dim=2)
    # [B, N]
    avg_k_dist = sum_k_dists / k.view(B, 1).float()

    # Sum these averages only for the unvisited nodes.
    avg_k_dist.masked_fill_(~unvisited_mask, 0.0)
    # [B]
    subtour_cost_estimate = avg_k_dist.sum(dim=1)

    # 2. Cost to connect the current node to the unvisited set.
    # [B, N]
    dist_from_current = torch.gather(dist_matrix, 1, current_node.view(B, 1, 1).expand(-1, 1, N)).squeeze(1)
    dist_from_current.masked_fill_(~unvisited_mask, float('inf'))
    # [B]
    min_dist_to_unvisited = torch.min(dist_from_current, dim=1).values

    # 3. Cost to connect the start node to the unvisited set.
    # [B, N]
    dist_from_start = torch.gather(dist_matrix, 1, start_node.view(B, 1, 1).expand(-1, 1, N)).squeeze(1)
    dist_from_start.masked_fill_(~unvisited_mask, float('inf'))
    # [B]
    min_dist_from_start = torch.min(dist_from_start, dim=1).values

    # Total estimated cost
    value = subtour_cost_estimate + min_dist_to_unvisited + min_dist_from_start

    # For terminal states, future cost is 0.
    value.masked_fill_(is_terminal, 0.0)
    # Replace any remaining infs (e.g., from min over empty set) with 0.
    value = torch.nan_to_num(value, posinf=0.0)

    return value.unsqueeze(-1)