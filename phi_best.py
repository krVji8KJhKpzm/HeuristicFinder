# THOUGHT: {use the average cosine of angles between vectors from current node to unvisited nodes as a spatial coherence score to scale remaining steps, providing a node-count-invariant heuristic}
def phi(state):
    B, N = state.action_mask().shape
    unvisited = state.unvisited_mask()                      # [B, N]
    coords = state.all_node_coords()                        # [B, N, 2]
    curr_idx = state.current_node_index()                   # [B]
    curr_coord = coords[torch.arange(B), curr_idx, :].unsqueeze(1)  # [B, 1, 2]
    # vectors from current to unvisited nodes
    vecs = coords - curr_coord                              # [B, N, 2]
    vecs = vecs * unvisited.unsqueeze(2).float()            # mask visited
    # normalize vectors
    norms = vecs.norm(dim=2, keepdim=True) + 1e-6
    unit_vecs = vecs / norms                                # [B, N, 2]
    # compute pairwise cosine similarities among unvisited
    cos = torch.bmm(unit_vecs, unit_vecs.transpose(1, 2))   # [B, N, N]
    cos = cos * (unvisited.unsqueeze(1) * unvisited.unsqueeze(2)).float()
    tri = torch.triu(torch.ones(N, N, device=cos.device), diagonal=1).bool()
    cos_tri = cos * tri.unsqueeze(0)
    avg_cos = cos_tri.sum(dim=(1, 2)) / (tri.unsqueeze(0).float().sum(dim=(1, 2)) * (unvisited.sum(dim=1).float() - 1).clamp(min=1) + 1e-6)
    coherence = 1.0 - avg_cos                               # high coherence -> low value
    # remaining steps
    rem = unvisited.sum(dim=1).float()
    # average unvisited distance to current
    dist = state.distance_matrix()                          # [B, N, N]
    curr_dists = dist[torch.arange(B), curr_idx, :]         # [B, N]
    avg_dist = (curr_dists * unvisited.float()).sum(dim=1) / (unvisited.sum(dim=1).float() + 1e-6)
    value = rem * avg_dist * (1.0 + 0.3 * coherence)
    scale = torch.sqrt(torch.tensor(N, dtype=torch.float32, device=value.device))
    return (value / scale).unsqueeze(1)                     # [B, 1]