# score=0.000000
# gamma=-0.100000
# code_hash=bf51f106ae4c9b9115e3047c69f3cb74b2253a9632a8105f4b09b574d11dec31
# THOUGHT: {auto}
# THOUGHT: {auto}
def phi(state):
    uv = state.unvisited_mask()                               # [B,N]
    B, N = uv.shape
    D = state.distance_matrix()                               # [B,N,N]
    # mask self and visited nodes
    D_uv = D.masked_fill(~uv.unsqueeze(1), 1e8)               # [B,N,N]
    # nearest unvisited neighbor per node
    nn_dst, _ = D_uv.min(dim=-1)                              # [B,N]
    # average nearest-neighbor distance over unvisited nodes
    avg_nn = (nn_dst * uv).sum(dim=-1) / (uv.sum(dim=-1) + 1e-6)  # [B]
    # symmetric return leg: distance from last unvisited to start
    first_idx = state.first_node_index()                      # [B]
    last_uv_idx = uv.max(dim=-1)[1]                           # [B] last unvisited index
    ret_leg = D[torch.arange(B), last_uv_idx, first_idx]      # [B]
    # estimate remaining length
    rem_est = avg_nn * uv.sum(dim=-1) + ret_leg               # [B]
    # add current tour length
    cur_len = D.gather(1, state.partial_path_indices().unsqueeze(-1).expand(-1,-1,2))
    cur_len = cur_len[:,:-1,0].gather(1, (state.partial_path_indices()!=-1).sum(dim=-1,keepdim=True)-1).squeeze(-1)
    cur_len = cur_len.where(cur_len>0, torch.zeros_like(cur_len))
    total_est = rem_est + cur_len                             # [B]
    return total_est.unsqueeze(-1)                            # [B,1]