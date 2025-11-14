# score=0.000000
# gamma=-0.100000
# code_hash=a8f2fb51fc819f52ed4efd2ba4c1a9e7e3dee65f3e6e2fba853f0889ddae55f0
# THOUGHT: {auto}
# THOUGHT: {auto}
def phi(state):
    # Extract coordinates and masks
    coords = state.all_node_coords()  # [B, N, 2]
    unvisited = state.unvisited_mask()  # [B, N]
    current_idx = state.current_node_index()  # [B]
    start_idx = state.first_node_index()  # [B]
    B, N, _ = coords.shape
    
    # Current and start coordinates
    current_coord = coords[torch.arange(B), current_idx, :]  # [B, 2]
    start_coord = coords[torch.arange(B), start_idx, :]  # [B, 2]
    
    # Compute pairwise distances among all nodes
    dist_all = state.distance_matrix()  # [B, N, N]
    
    # Mask distances for unvisited nodes only
    unvisited_f = unvisited.float().unsqueeze(1)  # [B, 1, N]
    dist_unvisited = dist_all * unvisited_f * unvisited_f.transpose(1, 2)  # [B, N, N]
    # Avoid zero diagonal to prevent softmin collapse
    dist_unvisited = dist_unvisited + (1 - unvisited_f) * 1e6  # mask out visited
    
    # Estimate MST length on unvisited nodes using softmin approximation of sum of smallest edges
    # Approximate MST as sum of smallest (N_unvisited - 1) edges connecting unvisited nodes
    # Use softmin to select small edges smoothly
    k = unvisited.sum(dim=1, keepdim=True) - 1  # [B, 1], number of edges in MST
    k = torch.clamp(k, min=0)
    
    # Flatten upper triangle (no self loops)
    tri_mask = torch.triu(torch.ones(N, N, device=coords.device, dtype=torch.bool), diagonal=1).unsqueeze(0)  # [1, N, N]
    flat_dist = torch.where(tri_mask, dist_unvisited, torch.full_like(dist_unvisited, 1e6))  # [B, N, N]
    flat_dist = flat_dist.view(B, -1)  # [B, N*(N-1)/2]
    
    # Softmin to approximate smallest k edges
    tau = 0.1
    weights = torch.softmax(-flat_dist / tau, dim=1)  # [B, N*(N-1)/2]
    sorted_vals, _ = torch.sort(flat_dist, dim=1)
    k_int = k.squeeze(1).long()
    mst_approx = torch.zeros(B, device=coords.device)
    for b in range(B):
        if k_int[b] > 0:
            mst_approx[b] = sorted_vals[b, :k_int[b]].sum()
    
    # Add distance from current to nearest unvisited
    dist_from_current = dist_all[torch.arange(B), current_idx, :]  # [B, N]
    dist_from_current = torch.where(unvisited, dist_from_current, torch.full_like(dist_from_current, 1e6))
    nearest_unvisited_dist = torch.min(dist_from_current, dim=1)[0]  # [B]
    
    # Add distance from last unvisited back to start (approximated as min distance from any unvisited to start)
    dist_to_start = dist_all[torch.arange(B), start_idx, :]  # [B, N]
    dist_to_start = torch.where(unvisited, dist_to_start, torch.full_like(dist_to_start, 1e6))
    min_dist_to_start = torch.min(dist_to_start, dim=1)[0]  # [B]
    
    # Total estimate
    total = mst_approx + nearest_unvisited_dist + min_dist_to_start  # [B]
    return total.unsqueeze(1)  # [B, 1]