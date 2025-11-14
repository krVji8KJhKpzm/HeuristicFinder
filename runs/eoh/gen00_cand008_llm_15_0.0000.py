# score=0.000000
# gamma=-0.100000
# code_hash=e1672a8e0a1db77f339615197b5a3b6853fc0d2710ce27f1150d322abf6276f6
# THOUGHT: {auto}
# THOUGHT: {auto}
def phi(state):
    coords = state.all_node_coords()  # [B, N, 2]
    unvisited = state.unvisited_mask()  # [B, N]
    current_idx = state.current_node_index()  # [B]
    B, N, _ = coords.shape
    # Get current node coordinates
    cur_coords = coords[torch.arange(B), current_idx].unsqueeze(1)  # [B, 1, 2]
    # Compute distances from current to all nodes
    dists = torch.norm(coords - cur_coords, p=2, dim=-1)  # [B, N]
    # Mask distances to unvisited nodes
    valid_dists = torch.where(unvisited, dists, torch.tensor(float('nan')).to(dists.device))
    # Compute mean distance over unvisited nodes
    mean_dist = torch.nanmean(valid_dists, dim=1, keepdim=True)  # [B, 1]
    # Scale to approximate expected remaining tour length
    return mean_dist * unvisited.float().sum(dim=1, keepdim=True)  # [B, 1]