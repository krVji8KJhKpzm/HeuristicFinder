# score=0.000000
# gamma=0.100000
# code_hash=fe575c25bebfa742a8461c7e957d1e5f05b9b4b400f63a67bfec97d5a5414434
# THOUGHT: {auto}
# THOUGHT: {auto}
def phi(state):
    # Get coordinates and masks
    coords = state.all_node_coords()  # [B,N,2]
    unvis = state.unvisited_mask()    # [B,N]
    cur_idx = state.current_node_index()  # [B]
    batch, N, _ = coords.size()
    
    # Coordinates of visited nodes are masked to zero so they don't affect reductions
    coords_unvis = coords * unvis.unsqueeze(-1)  # [B,N,2]
    
    # Pairwise distances among unvisited nodes
    # Compute pairwise differences
    dx = coords_unvis.unsqueeze(2) - coords_unvis.unsqueeze(1)  # [B,N,N,2]
    dist_mat = torch.norm(dx, p=2, dim=-1)  # [B,N,N]
    # Zero out diagonal and pairs involving visited nodes
    unvis_pair = unvis.unsqueeze(1) * unvis.unsqueeze(2)  # [B,N,N]
    dist_mat = dist_mat * unvis_pair
    # Count of unvisited nodes per batch
    n_unvis = unvis.sum(dim=1, keepdim=True).clamp(min=1)  # [B,1]
    # Average pairwise distance among unvisited
    avg_pair_dist = dist_mat.sum(dim=(1,2), keepdim=True) / (n_unvis * (n_unvis - 1)).clamp(min=1)
    
    # Distance from current node to nearest unvisited
    cur_coord = coords[torch.arange(batch), cur_idx, :]  # [B,2]
    dist_to_cur = torch.norm(coords_unvis - cur_coord.unsqueeze(1), p=2, dim=-1)  # [B,N]
    dist_to_cur = dist_to_cur * unvis  # mask visited
    min_dist_to_cur, _ = dist_to_cur.min(dim=1, keepdim=True)  # [B,1]
    
    # Combine: avg pairwise distance + nearest distance as heuristic for remaining tour
    heuristic = avg_pair_dist + min_dist_to_cur
    return heuristic