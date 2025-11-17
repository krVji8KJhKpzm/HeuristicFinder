# score=0.180603
# gamma=-0.100000
# code_hash=f4046de7de2b1d04b827c811d4bd9c27f6c48154a814762acb33f2134cabbab6
# stats: mse=5.537; rmse=2.35308; mse_tsp100=21.0141; mse_tsp20=5.537; mse_tsp50=11.5361; rmse_tsp100=4.58411; rmse_tsp20=2.35308; rmse_tsp50=3.39649
# ALGORITHM: {auto} def phi(state): """ {Estimate future tour length by finding the minimum distance from each unvisited node to any other available node (unvisited or current), summing these minimums, and scaling by the log of remaining nodes.}
# THOUGHT: {auto}
def phi(state):
    """
    {Estimate future tour length by finding the minimum distance from each unvisited node to any other available node (unvisited or current), summing these minimums, and scaling by the log of remaining nodes.}
    Args:
        state: TSPStateView with batch dimension B and N nodes.
    Returns:
        A tensor of shape [B, 1] representing the estimated future cost (negative value).
    """
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    # [B, 1, N]
    unvisited_mask_from = unvisited_mask.unsqueeze(1)
    # [B, N, 1]
    unvisited_mask_to = unvisited_mask.unsqueeze(2)
    # [B, N]
    current_node = state.current_node_index()
    # [B, N]
    current_mask = torch.zeros_like(unvisited_mask).scatter_(1, current_node.unsqueeze(1), 1)
    # [B, N, 1]
    current_mask_to = current_mask.unsqueeze(2)
    
    # [B, N, 1] mask for valid 'to' nodes (unvisited or current)
    # This allows connecting from an unvisited node to any other unvisited node or back to the current one.
    available_to_mask = unvisited_mask_to | current_mask_to
    
    # [B, N, N] mask for valid connections: from unvisited to available
    valid_connections_mask = unvisited_mask_from & available_to_mask
    
    # Set distances for invalid connections to a large value
    large_value = 1e9
    masked_dist = torch.where(valid_connections_mask, dist_matrix, large_value)
    
    # For each unvisited node ('from' node), find the minimum distance to any available 'to' node
    # [B, N]
    min_dist_from_unvisited, _ = torch.min(masked_dist, dim=2)
    
    # Sum these minimum distances, but only for the unvisited nodes.
    # We set the distances for already visited nodes to 0 so they don't contribute to the sum.
    min_dist_from_unvisited = torch.where(unvisited_mask, min_dist_from_unvisited, 0.0)
    # [B]
    sum_of_min_dists = torch.sum(min_dist_from_unvisited, dim=1)
    
    # [B]
    num_unvisited = torch.sum(unvisited_mask, dim=1, dtype=torch.float32)
    
    # Scale the sum by a factor related to the number of remaining nodes.
    # Using log helps to moderate the scaling effect as the tour progresses.
    # Add 1.0 to avoid log(0) and log(1)=0 which would zero out the potential.
    # Adding 1.0 to num_unvisited ensures the scaler is > 0.
    scaler = torch.log1p(num_unvisited)
    
    # The potential is the scaled sum of minimum distances.
    # It represents an optimistic estimate of the remaining path length.
    # [B]
    potential = sum_of_min_dists * scaler
    
    # The value function V is the negative of the cost-to-go.
    # Since phi approximates cost-to-go, we return its negative.
    # [B, 1]
    value = -potential.unsqueeze(1)
    
    return value