# score=0.000000
# gamma=-0.100000
# code_hash=082ac34a0d5fa66ebe3a524d09a3f491e7ef485b2f030f0eed509652e18704b6
# THOUGHT: {auto}
# THOUGHT: {auto}
def phi(state):
    unvisited = state.unvisited_mask()  # [B,N]
    num_unvisited = unvisited.sum(dim=1, keepdim=True).float()  # [B,1]
    current_idx = state.current_node_index()  # [B]
    start_idx = state.first_node_index()  # [B]
    dist = state.distance_matrix()  # [B,N,N]
    
    # Gather distances from current node to all nodes
    cur_dists = dist[torch.arange(dist.size(0)), current_idx]  # [B,N]
    # Mask to unvisited nodes only
    cur_to_unvisited = torch.where(unvisited, cur_dists, torch.tensor(float('inf'), device=dist.device))
    avg_dist_from_current = (cur_to_unvisited.sum(dim=1, keepdim=True) / num_unvisited)  # [B,1]
    
    # Nearest unvisited to start
    start_dists = dist[torch.arange(dist.size(0)), start_idx]  # [B,N]
    start_to_unvisited = torch.where(unvisited, start_dists, torch.tensor(float('inf'), device=dist.device))
    min_back_dist = start_to_unvisited.min(dim=1, keepdim=True)[0]  # [B,1]
    
    # Estimate remaining length
    remaining = num_unvisited * avg_dist_from_current + min_back_dist
    return remaining.unsqueeze(1)  # Ensure [B,1]