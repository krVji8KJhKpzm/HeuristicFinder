# score=0.000000
# gamma=1.000000
# code_hash=84dc0532ce6459298f55b24ce7843f8ca10f57accf6446858c4f8f3d1380f112
# THOUGHT: {auto}
# THOUGHT: {auto}
def phi(state):
    unvisited = state.unvisited_mask()                     # [B, N]
    coords = state.all_node_coords()                       # [B, N, 2]
    dist = state.distance_matrix()                         # [B, N, N]
    B, N = unvisited.shape

    # Compute pairwise distances among unvisited nodes
    unvisited_f = unvisited.float().unsqueeze(-1)          # [B, N, 1]
    # Mask distances: zero out rows/cols of visited nodes
    masked_dist = dist * unvisited_f.unsqueeze(1) * unvisited_f.unsqueeze(2)  # [B, N, N]
    # Count unvisited pairs per batch
    counts = unvisited.sum(dim=1, keepdim=True)            # [B, 1]
    valid_pairs = counts * (counts - 1) + 1e-6             # avoid div0
    avg_pairwise = (masked_dist.sum(dim=(1,2), keepdim=True) / valid_pairs).squeeze(-1)  # [B, 1]

    # Distance from current location to nearest unvisited node
    cur_idx = state.current_node_index()                   # [B]
    cur_dists = dist[torch.arange(B), cur_idx]             # [B, N]
    cur_dists = cur_dists.masked_fill(~unvisited, float('inf'))
    min_cur = cur_dists.min(dim=1, keepdim=True)[0]        # [B, 1]

    # Distance from start location to nearest unvisited node
    start_idx = state.first_node_index()                   # [B]
    start_dists = dist[torch.arange(B), start_idx]         # [B, N]
    start_dists = start_dists.masked_fill(~unvisited, float('inf'))
    min_start = start_dists.min(dim=1, keepdim=True)[0]    # [B, 1]

    # Combine components and normalize by unvisited count
    phi_val = (avg_pairwise + min_cur + min_start) / (counts.unsqueeze(-1) + 1)
    return phi_val