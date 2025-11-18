# score=0.002822
# gamma=1.000000
# code_hash=20b90a53a5ad12b5e48dade46876d784234cacd827017f98aa9fb4ff37d2e088
# stats: mse=9.33898; rmse=3.05597; mse_tsp100=354.331; mse_tsp20=9.33898; mse_tsp50=79.2363; mse_worst=354.331; rmse_tsp100=18.8237; rmse_tsp20=3.05597; rmse_tsp50=8.90148; rmse_worst=18.8237
# ALGORITHM: {auto} def phi(state): """ Estimates the future tour length (cost-to-go) for the TSP. Args: state (TSPStateView): The current state of the TSP environment. Returns: torch.Tensor: A scalar tensor of shape [B, 1] representing the estimated future tour length. """ # {The algorithm estimates future tour length by combining three components: the average distance from the current node to unvisited nodes, the average distance from the start node back to unvisited nodes, and the expected path length within the remaining unvisited nodes based on their average pairwise distance.}
# THOUGHT: {auto}
def phi(state):
    """
    Estimates the future tour length (cost-to-go) for the TSP.

    Args:
        state (TSPStateView): The current state of the TSP environment.

    Returns:
        torch.Tensor: A scalar tensor of shape [B, 1] representing the estimated future tour length.
    """
    # {The algorithm estimates future tour length by combining three components: the average distance from the current node to unvisited nodes, the average distance from the start node back to unvisited nodes, and the expected path length within the remaining unvisited nodes based on their average pairwise distance.}
    
    # Get state information
    dist_matrix = state.distance_matrix()  # [B, N, N]
    unvisited_mask = state.unvisited_mask()  # [B, N]
    current_node = state.current_node_index()  # [B]
    first_node = state.first_node_index()  # [B]
    
    # Useful dimensions and counts
    B, N = dist_matrix.shape[:2]
    device = dist_matrix.device
    
    # Number of unvisited nodes, handle division by zero
    num_unvisited = unvisited_mask.sum(dim=1, keepdim=True).float()  # [B, 1]
    is_done = (num_unvisited <= 1e-9) # [B,1]

    # 1. Cost from current node to the next unvisited node
    # Gather distances from the current node to all other nodes
    current_idx_exp = current_node.view(B, 1, 1).expand(-1, 1, N) # [B,1,N]
    dist_from_current = torch.gather(dist_matrix, 1, current_idx_exp).squeeze(1)  # [B, N]
    
    # Mask to keep only distances to unvisited nodes
    dist_from_current_to_unvisited = dist_from_current * unvisited_mask
    
    # Average distance from current to unvisited (cost to connect to the remaining subgraph)
    # Using a safe division
    sum_dist_curr_unvisited = dist_from_current_to_unvisited.sum(dim=1, keepdim=True) # [B,1]
    avg_dist_curr_unvisited = sum_dist_curr_unvisited / torch.clamp(num_unvisited, min=1.0) # [B,1]
    
    # 2. Cost from the last unvisited node back to the start node
    # Gather distances from the start node to all other nodes
    first_idx_exp = first_node.view(B, 1, 1).expand(-1, 1, N) # [B,1,N]
    dist_from_start = torch.gather(dist_matrix, 1, first_idx_exp).squeeze(1) # [B, N]
    
    # Mask to keep only distances to unvisited nodes
    dist_from_start_to_unvisited = dist_from_start * unvisited_mask
    
    # Average distance from start to unvisited (cost to close the loop)
    sum_dist_start_unvisited = dist_from_start_to_unvisited.sum(dim=1, keepdim=True) # [B,1]
    avg_dist_start_unvisited = sum_dist_start_unvisited / torch.clamp(num_unvisited, min=1.0) # [B,1]

    # 3. Estimated cost of the tour within the unvisited nodes
    # Create a mask for pairs of unvisited nodes
    unvisited_mask_2d = unvisited_mask.unsqueeze(2) * unvisited_mask.unsqueeze(1)  # [B, N, N]
    
    # Sum of distances between all pairs of unvisited nodes
    sum_dist_unvisited_pairs = (dist_matrix * unvisited_mask_2d).sum(dim=(1, 2), keepdim=True) # [B,1,1]
    sum_dist_unvisited_pairs = sum_dist_unvisited_pairs.squeeze(-1) # [B,1]
    
    # Number of pairs of unvisited nodes
    num_pairs = num_unvisited * (num_unvisited - 1) # [B,1]
    
    # Average distance between any two unvisited nodes
    avg_inter_unvisited_dist = sum_dist_unvisited_pairs / torch.clamp(num_pairs, min=1.0) # [B,1]
    
    # The MST/tour length is roughly proportional to (N-1) * avg_edge_length
    # We use (num_unvisited - 1) as the scaling factor.
    internal_tour_cost = avg_inter_unvisited_dist * torch.clamp(num_unvisited - 1, min=0.0) # [B,1]

    # Combine the three components
    # The total future cost is the sum of these three estimates.
    # We use a small scaling factor on the internal cost to keep it moderate.
    value = avg_dist_curr_unvisited + avg_dist_start_unvisited + 0.8 * internal_tour_cost # [B,1]
    
    # For terminal states, the future cost is zero.
    final_value = torch.where(is_done, torch.zeros_like(value), value)
    
    return final_value