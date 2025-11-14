# score=0.000000
# gamma=0.100000
# code_hash=d0d85f46ba703cb161d46768b48e2090606e10461ef7e6a2a5571060f70bacfd
# THOUGHT: {auto}
# THOUGHT: {auto}
def phi(state):
    coords = state.all_node_coords()  # [B,N,2]
    unvisited = state.unvisited_mask()  # [B,N]
    start_idx = state.first_node_index()  # [B]
    B, N, _ = coords.shape
    
    # Compute pairwise distances
    delta = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B,N,N,2]
    dist = torch.norm(delta, dim=-1)  # [B,N,N]
    
    # Mask out visited nodes and self-loops (set large distance)
    mask = unvisited.unsqueeze(1) & (torch.arange(N, device=coords.device) != torch.arange(N, device=coords.device).view(1, N, 1))
    masked_dist = torch.where(mask, dist, torch.full_like(dist, 1e8))
    
    # Nearest unvisited neighbor per node (only among unvisited)
    nearest_dist, _ = masked_dist.min(dim=-1)  # [B,N]
    # Only consider unvisited nodes
    nearest_unvisited = torch.where(unvisited, nearest_dist, torch.zeros_like(nearest_dist))
    avg_nearest = nearest_unvisited.sum(dim=1) / unvisited.sum(dim=1).clamp(min=1)  # [B]
    
    # Distance from current node to start (for return leg)
    cur_idx = state.current_node_index()  # [B]
    batch_idx = torch.arange(B, device=coords.device)
    cur2start = dist[batch_idx, cur_idx, start_idx]  # [B]
    
    # Estimate remaining tour: unvisited nodes * avg step + return
    remain = unvisited.sum(dim=1).float() * avg_nearest + cur2start  # [B]
    return remain.unsqueeze(1)  # [B,1]