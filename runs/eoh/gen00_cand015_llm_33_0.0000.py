# score=0.000000
# gamma=0.100000
# code_hash=4b539f497329650ce2ddd42da04ec72d3d5110269a8cdd1799541927e804e755
# THOUGHT: {auto}
# THOUGHT: {auto}
def phi(state):
    unvisited = state.unvisited_mask().float()                       # [B,N]
    num_unvisited = unvisited.sum(dim=1, keepdim=True)             # [B,1]
    D = state.distance_matrix()                                    # [B,N,N]
    # average distance from each node to all others (mask self-distance=0)
    avg_dist = (D.sum(dim=2) / (D.size(1) - 1)).unsqueeze(2)       # [B,N,1]
    # pick unvisited nodes' avg distances
    unvisited_avg = (unvisited.unsqueeze(2) * avg_dist).sum(dim=1)  # [B,1]
    # reduce over batch to keep node-count invariance
    est = unvisited_avg * num_unvisited * 0.05                     # [B,1]
    return est.clamp(min=0.0)