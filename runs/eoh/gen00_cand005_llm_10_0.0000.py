# score=0.000000
# gamma=-0.100000
# code_hash=47fe4b4935ba1cdfeb08b631bc6a12bf23744b8f5d5c789f58e3a5f163b585b4
# THOUGHT: {auto}
# THOUGHT: {auto}
def phi(state):
    unvisited = state.unvisited_mask()                      # [B,N]
    coords = state.all_node_coords()                        # [B,N,2]
    B, N, _ = coords.shape
    
    # Pairwise distances: [B,N,N]
    delta = coords.unsqueeze(2) - coords.unsqueeze(1)
    dist = (delta ** 2).sum(-1).sqrt()
    
    # Mask out visited nodes for nearest-neighbor search
    masked = dist.masked_fill(~unvisited.unsqueeze(1), float('inf'))
    
    # Nearest unvisited neighbor distance per node (inf for visited)
    nearest, _ = masked.min(dim=-1)                         # [B,N]
    nearest = nearest.masked_fill(~unvisited, 0.0)
    
    # Average nearest-neighbor distance among unvisited
    avg_nn = nearest.sum(dim=-1) / unvisited.sum(dim=-1).clamp(min=1)  # [B]
    
    # Distance from each unvisited to start
    start_idx = state.first_node_index()                    # [B]
    start_coord = coords.gather(1, start_idx.view(B,1,1).expand(B,1,2)).squeeze(1)  # [B,2]
    delta_start = coords - start_coord.unsqueeze(1)         # [B,N,2]
    dist_start = (delta_start ** 2).sum(-1).sqrt()          # [B,N]
    dist_start = dist_start.masked_fill(~unvisited, 0.0)
    
    # Minimum distance back to start among unvisited
    min_back, _ = dist_start.max(dim=-1)                    # [B] (use max to avoid 0 when few left)
    
    # Remaining length estimate
    n_unvisited = unvisited.sum(dim=-1).float()             # [B]
    remaining = avg_nn * n_unvisited + min_back
    
    # Normalize by sqrt(N) for scale invariance
    scale = (N ** 0.5).float()
    return (remaining / scale).unsqueeze(-1)                # [B,1]