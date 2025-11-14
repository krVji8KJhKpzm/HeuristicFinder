# score=0.000000
# gamma=1.000000
# code_hash=b1a801b755d1cf440bedb80ef5f35e4d97f998f8ecfd3daf568f4932ae8511ba
# THOUGHT: {auto}
# THOUGHT: {auto}
def phi(state):
    unvisited = state.unvisited_mask()                       # [B,N]
    coords = state.all_node_coords()                         # [B,N,2]
    B, N, _ = coords.shape

    # pairwise distances
    d = torch.cdist(coords, coords, p=2)                     # [B,N,N]

    # mask self and visited
    d = d.masked_fill(~unvisited.unsqueeze(1), 1e8)          # mask visited rows
    d = d.masked_fill(~unvisited.unsqueeze(2), 1e8)          # mask visited cols

    # nearest unvisited neighbor per node (still unvisited)
    nn, _ = d.min(dim=-1)                                    # [B,N]
    nn = nn.masked_fill(~unvisited, 0.0)                     # zero for visited

    # avg nearest-neighbor distance among unvisited
    avg_nn = nn.sum(dim=-1) / unvisited.sum(dim=-1).clamp(min=1)  # [B]

    # distance back to start
    start_idx = state.first_node_index()                     # [B]
    start_coord = coords[torch.arange(B), start_idx]         # [B,2]
    cur_idx = state.current_node_index()                     # [B]
    cur_coord = coords[torch.arange(B), cur_idx]             # [B,2]
    return_to_start = (cur_coord - start_coord).norm(p=2, dim=-1)  # [B]

    # remaining nodes
    rem = unvisited.sum(dim=-1).float()                      # [B]

    # heuristic: rem * avg_nn + return leg
    est = rem * avg_nn + return_to_start                     # [B]
    return est.unsqueeze(-1)                                 # [B,1]