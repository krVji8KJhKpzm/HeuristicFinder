# score=0.000000
# gamma=1.000000
# code_hash=5e5f92ae5bf49c7b7a1a8042550ded0bb66616d7b151df7bb688eafd994032c9
# THOUGHT: {auto}
# THOUGHT: {auto}
def phi(state):
    coords = state.all_node_coords()  # [B, N, 2]
    unvisited = state.unvisited_mask()  # [B, N]
    current_idx = state.current_node_index()  # [B]
    B, N, _ = coords.size()
    
    # Current node coordinates
    current_coord = coords[torch.arange(B), current_idx].unsqueeze(1)  # [B, 1, 2]
    
    # Unvisited node coordinates
    unvisited_coords = coords * unvisited.unsqueeze(-1)  # [B, N, 2]
    unvisited_counts = unvisited.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
    
    # Average distance from current node to unvisited nodes
    dist_to_unvisited = torch.norm(unvisited_coords - current_coord, dim=2)  # [B, N]
    avg_dist_to_unvisited = (dist_to_unvisited * unvisited).sum(dim=1, keepdim=True) / unvisited_counts  # [B, 1]
    
    # Pairwise distances among unvisited nodes
    unvisited_coords_expanded = unvisited_coords.unsqueeze(2)  # [B, N, 1, 2]
    unvisited_coords_tiled = unvisited_coords.unsqueeze(1)  # [B, 1, N, 2]
    pairwise_dists = torch.norm(unvisited_coords_expanded - unvisited_coords_tiled, dim=3)  # [B, N, N]
    pairwise_mask = unvisited.unsqueeze(2) * unvisited.unsqueeze(1)  # [B, N, N]
    pairwise_dists = pairwise_dists * pairwise_mask  # Mask out non-unvisited pairs
    pairwise_counts = pairwise_mask.sum(dim=(1, 2), keepdim=True).clamp(min=1)  # [B, 1]
    avg_pairwise_dist = pairwise_dists.sum(dim=(1, 2), keepdim=True) / pairwise_counts  # [B, 1]
    
    # Combine both terms as an estimate of remaining tour length
    remaining_estimate = avg_dist_to_unvisited + avg_pairwise_dist
    return remaining_estimate  # [B, 1]