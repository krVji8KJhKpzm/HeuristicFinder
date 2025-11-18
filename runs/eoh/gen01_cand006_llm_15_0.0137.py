# score=0.013693
# gamma=0.100000
# code_hash=c56b73843f678df98a303431c2ce030b166c3f27684f9e1488fd86edf87774e0
# stats: mse=22.5157; rmse=4.74507; mse_tsp100=73.0308; mse_tsp20=22.5157; mse_tsp50=42.5038; mse_worst=73.0308; rmse_tsp100=8.54581; rmse_tsp20=4.74507; rmse_tsp50=6.51949; rmse_worst=8.54581
# ALGORITHM: {auto}
# THOUGHT: {auto}
def phi(state):
    """
    Estimates the future tour length (cost-to-go) for the TSP.
    The value is composed of three main parts:
    1. The expected distance from the current node to the next unvisited node.
    2. The expected total distance for the tour among the remaining unvisited nodes.
    3. The expected distance from the last unvisited node back to the start node.
    This provides a heuristic for the remaining path length.

    Args:
        state (TSPStateView): The current state of the TSP environment.

    Returns:
        torch.Tensor: A scalar tensor [B, 1] representing the estimated future cost.
    """
    # Get batch size and number of nodes
    B, N, _ = state.all_node_coords().shape
    device = state.all_node_coords().device

    # Get masks for visited and unvisited nodes
    unvisited_mask = state.unvisited_mask()  # [B, N]
    visited_mask = state.visited_mask()      # [B, N]

    # Count the number of unvisited nodes
    num_unvisited = unvisited_mask.sum(dim=1, keepdim=True)  # [B, 1]
    is_done = (num_unvisited <= 1) # If 0 or 1 unvisited, the tour is effectively done or in the last step.

    # 1. Distance from the current node to all unvisited nodes
    dist_matrix = state.distance_matrix()  # [B, N, N]
    current_node_idx = state.current_node_index().view(B, 1, 1).expand(-1, 1, N) # [B, 1, N]
    dist_from_current = torch.gather(dist_matrix, 1, current_node_idx).squeeze(1) # [B, N]

    # Mask out distances to already visited nodes
    dist_from_current_to_unvisited = dist_from_current.masked_fill(visited_mask, float('inf'))
    
    # Heuristic for the next step: average distance to an unvisited node
    # Use a small epsilon to avoid division by zero
    avg_dist_to_next = dist_from_current_to_unvisited.nan_to_num(posinf=0.0).sum(dim=1, keepdim=True) / torch.clamp(num_unvisited, min=1.0)

    # 2. Heuristic for the tour among remaining unvisited nodes
    # For each unvisited node, find its minimum distance to another unvisited node
    unvisited_to_unvisited_mask = unvisited_mask.unsqueeze(2) & unvisited_mask.unsqueeze(1) # [B, N, N]
    dist_unvisited = dist_matrix.masked_fill(~unvisited_to_unvisited_mask, float('inf'))
    
    # Set diagonal to infinity to find min distance to *another* unvisited node
    dist_unvisited.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))
    
    # Find the minimum distance from each unvisited node to any other unvisited node
    min_dist_per_unvisited_node, _ = torch.min(dist_unvisited, dim=2) # [B, N]
    
    # Sum these minimum distances and average them to get a per-node cost estimate
    total_min_dist = min_dist_per_unvisited_node.nan_to_num(posinf=0.0).sum(dim=1, keepdim=True) # [B, 1]
    avg_min_dist_unvisited = total_min_dist / torch.clamp(num_unvisited, min=1.0)
    
    # Scale by the number of remaining steps (num_unvisited - 1)
    # This approximates a nearest-neighbor heuristic for the remaining sub-tour
    remaining_subtour_cost = avg_min_dist_unvisited * torch.clamp(num_unvisited - 1, min=0.0)

    # 3. Heuristic for returning to the start node from the last unvisited node
    start_node_idx = state.first_node_index().view(B, 1, 1).expand(-1, 1, N) # [B, 1, N]
    dist_to_start = torch.gather(dist_matrix, 1, start_node_idx).squeeze(1) # [B, N]
    
    # Average distance from any unvisited node back to the start
    dist_unvisited_to_start = dist_to_start.masked_fill(visited_mask, 0.0)
    avg_dist_to_start = dist_unvisited_to_start.sum(dim=1, keepdim=True) / torch.clamp(num_unvisited, min=1.0)
    
    # Combine the components
    # The total estimated future cost is the sum of the three parts.
    # We use num_unvisited > 1 as a condition for adding the subtour and start-return costs.
    # When num_unvisited == 1, the only remaining action is to return to start.
    # When num_unvisited == 0, the tour is done, cost is 0.
    
    # If only one node left, cost is just dist from current to it, plus dist from it to start
    last_step_mask = (num_unvisited == 1.0)
    last_unvisited_idx = unvisited_mask.long().argmax(dim=1, keepdim=True) # [B, 1]
    last_unvisited_idx_exp = last_unvisited_idx.unsqueeze(2) # [B, 1, 1]
    
    dist_curr_to_last = torch.gather(dist_from_current, 1, last_unvisited_idx) # [B, 1]
    dist_last_to_start = torch.gather(dist_to_start, 1, last_unvisited_idx) # [B, 1]
    last_step_cost = dist_curr_to_last + dist_last_to_start
    
    # Calculate the full heuristic
    value = avg_dist_to_next + remaining_subtour_cost + avg_dist_to_start
    
    # Apply special case for the last step
    value = torch.where(last_step_mask, last_step_cost, value)

    # If the episode is done (or about to be), the future cost is 0.
    # This ensures terminal consistency.
    value = value.masked_fill(is_done, 0.0)
    
    # Return as a negative value because the environment uses negative rewards (path length)
    return -value