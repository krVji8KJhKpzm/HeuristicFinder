# score=0.000000
# gamma=1.000000
# code_hash=8ba7f98f7274bcef85b8c3739a0ef7dd5a776d27646e4d40852f3cead98d2a4b
# THOUGHT: {auto}
# THOUGHT: {auto}
def phi(state):
    unvisited = state.unvisited_mask()                      # [B,N]
    N = unvisited.shape[1]
    idx = state.current_node_index()                        # [B]
    dist = state.distance_matrix()                          # [B,N,N]
    cur_d = dist[torch.arange(dist.shape[0]), idx]          # [B,N]
    start_d = dist[torch.arange(dist.shape[0]), state.first_node_index()]  # [B,N]
    # mask out visited nodes
    cur_d = cur_d.masked_fill(~unvisited, 0.0)
    start_d = start_d.masked_fill(~unvisited, 0.0)
    # mean distance to unvisited
    avg_cur = cur_d.sum(dim=1) / (unvisited.sum(dim=1).clamp_min(1))
    avg_start = start_d.sum(dim=1) / (unvisited.sum(dim=1).clamp_min(1))
    # heuristic: twice average radius plus closing leg
    rem = 2 * avg_cur * unvisited.sum(dim=1).float() + avg_start
    return rem.unsqueeze(1)                                 # [B,1]