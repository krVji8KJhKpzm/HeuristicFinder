# score=0.050303
# gamma=1.000000
# code_hash=a0d8b8cd0955fa9b8200849b52e3cc78956528631d7eb1a3016d0bf7085d320a
# stats: mse=4.66906; rmse=2.1608; mse_tsp100=19.8796; mse_tsp20=4.66906; mse_tsp50=10.5161; mse_worst=19.8796; rmse_tsp100=4.45865; rmse_tsp20=2.1608; rmse_tsp50=3.24285; rmse_worst=4.45865
# ALGORITHM: {auto} def phi(state): """ Estimates future tour length based on the convex hull of unvisited nodes, plus connection costs. {The algorithm estimates the remaining tour length by summing three components: the perimeter of the convex hull of the unvisited nodes, the minimum distance from the current node to a node on the hull, and the minimum distance from the starting node to a node on the hull if it's unvisited.}
# THOUGHT: {auto}
def phi(state):
    """
    Estimates future tour length based on the convex hull of unvisited nodes, plus connection costs.
    {The algorithm estimates the remaining tour length by summing three components: the perimeter of the convex hull of the unvisited nodes, the minimum distance from the current node to a node on the hull, and the minimum distance from the starting node to a node on the hull if it's unvisited.}

    Args:
        state: A TSPStateView object with batch-friendly helper methods.

    Returns:
        A torch tensor of shape [B, 1] representing the estimated future tour length.
    """
    # [B, N, 2]
    coords = state.all_node_coords()
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    # [B]
    current_node_idx = state.current_node_index()
    # [B]
    start_node_idx = state.first_node_index()
    # [B, N, N]
    dist_matrix = state.distance_matrix()

    B, N, _ = coords.shape
    device = coords.device

    # Handle terminal states where few or no nodes are unvisited
    n_unvisited = unvisited_mask.sum(dim=1)
    is_terminal = n_unvisited <= 2
    if is_terminal.all():
        return torch.zeros(B, 1, device=device)

    # 1. Calculate the convex hull perimeter of unvisited nodes
    # A large value to effectively remove visited nodes from consideration
    inf_val = coords.abs().max().item() * 1e3 if coords.numel() > 0 else 1e9
    
    # [B, N, 2], move visited nodes far away
    unvisited_coords = torch.where(unvisited_mask.unsqueeze(-1), coords, torch.full_like(coords, inf_val))

    # Find the bottom-left point (start of the hull) for each batch item
    # [B, 2]
    start_point_coords = torch.full_like(coords[:, 0, :], inf_val)
    start_point_coords[:, 1] = unvisited_coords[:, :, 1].min(dim=1).values
    # Break ties with the minimum x-coordinate
    y_min_mask = (unvisited_coords[:, :, 1] == start_point_coords[:, 1].unsqueeze(1))
    x_for_y_min = torch.where(y_min_mask, unvisited_coords[:, :, 0], torch.full_like(coords[:, 0, 0], inf_val))
    start_point_coords[:, 0] = x_for_y_min.min(dim=1).values

    # Calculate angles from the start point to all other unvisited points
    # [B, N, 2]
    vectors = unvisited_coords - start_point_coords.unsqueeze(1)
    # [B, N]
    angles = torch.atan2(vectors[..., 1], vectors[..., 0])
    # Set angle for start point itself to a large negative value to keep it first
    is_start_point = (unvisited_coords == start_point_coords.unsqueeze(1)).all(dim=-1)
    angles = torch.where(is_start_point, -torch.pi * 2, angles)
    # Set angles for visited points to a large value to sort them last
    angles = torch.where(unvisited_mask, angles, torch.pi * 2)

    # Sort indices by angle to get the hull path
    # [B, N]
    _, sorted_indices = torch.sort(angles, dim=1)
    # [B, N, 2]
    sorted_coords = torch.gather(coords, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, 2))
    
    # Calculate perimeter of the sorted points (which form the convex hull)
    # [B, N, 2]
    rolled_coords = torch.roll(sorted_coords, shifts=-1, dims=1)
    # [B, N]
    segment_lengths = torch.linalg.norm(sorted_coords - rolled_coords, dim=-1)
    # [B]
    hull_perimeter = torch.where(n_unvisited > 0, segment_lengths.sum(dim=1), torch.tensor(0.0, device=device))

    # 2. Minimum distance from current node to the convex hull (unvisited nodes)
    # [B, N]
    current_dists = torch.gather(dist_matrix, 1, current_node_idx.view(B, 1, 1).expand(-1, 1, N)).squeeze(1)
    current_dists_unvisited = torch.where(unvisited_mask, current_dists, torch.full_like(current_dists, float('inf')))
    # [B]
    min_dist_to_hull = current_dists_unvisited.min(dim=1).values
    # If no unvisited nodes, this will be inf; handle below.

    # 3. Minimum distance from start node to the convex hull (if start node is unvisited)
    # [B, N]
    start_dists = torch.gather(dist_matrix, 1, start_node_idx.view(B, 1, 1).expand(-1, 1, N)).squeeze(1)
    start_dists_unvisited = torch.where(unvisited_mask, start_dists, torch.full_like(start_dists, float('inf')))
    # [B]
    min_dist_from_start = start_dists_unvisited.min(dim=1).values
    
    # Check if start node has been visited
    # [B]
    start_node_visited_mask = state.visited_mask().gather(1, start_node_idx.unsqueeze(1)).squeeze(1)
    # Cost is zero if start node is already visited or is the current node
    start_connection_cost = torch.where(start_node_visited_mask | (current_node_idx == start_node_idx), 0.0, min_dist_from_start)
    
    # Combine the components
    # [B]
    total_value = hull_perimeter + min_dist_to_hull + start_connection_cost

    # Final cleanup for terminal states
    final_value = torch.where(is_terminal, 0.0, total_value)

    return final_value.unsqueeze(-1)