# score=0.000000
# gamma=-0.100000
# code_hash=101a2da6626d91df4a9e657ee71938c7f8202366bdd7818531107899ddf2c5a9
# THOUGHT: {auto}
# THOUGHT: {auto}
def phi(state):
    unvisited = state.unvisited_mask()                      # [B,N]
    num_unvisited = unvisited.sum(dim=1, keepdim=True)      # [B,1]
    dm = state.distance_matrix()                            # [B,N,N]
    # mask self and visited; inf for invalid
    masked_dm = dm.masked_fill(~unvisited.unsqueeze(1), float('inf'))
    # nearest unvisited neighbor distance per node
    nearest = masked_dm.min(dim=2).values                   # [B,N]
    # average over unvisited nodes
    avg_nearest = (nearest * unvisited).sum(dim=1, keepdim=True) / num_unvisited.clamp(min=1)
    # distance from current node to nearest unvisited
    cur_idx = state.current_node_index()                    # [B]
    cur_dist = nearest.gather(1, cur_idx.unsqueeze(1))      # [B,1]
    # heuristic: (num_unvisited * avg_nearest) + cur_dist
    return (num_unvisited * avg_nearest) + cur_dist         # [B,1]