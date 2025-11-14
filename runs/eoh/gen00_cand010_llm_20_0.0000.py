# score=0.000000
# gamma=0.100000
# code_hash=d387a1134ce935c96018531b02681c64cc83bf695ed840ee401b59782759111c
# THOUGHT: {auto}
# THOUGHT: {auto}
def phi(state):
    unvisited = state.unvisited_mask()                      # [B,N]
    visited = state.visited_mask()                          # [B,N]
    dm = state.distance_matrix()                            # [B,N,N]
    # min distance from each unvisited node to any visited node (or to start if none)
    min_to_visited, _ = (dm * visited.unsqueeze(1)).min(dim=2)  # [B,N]
    min_to_visited = min_to_visited.clamp_min(0.0)
    # take only unvisited nodes
    unvisited_sum = (min_to_visited * unvisited).sum(dim=1)     # [B]
    unvisited_cnt = unvisited.sum(dim=1).clamp_min(1)           # [B]
    avg_unvisited_cost = unvisited_sum / unvisited_cnt          # [B]
    # distance from current node to nearest unvisited and back to start
    cur_idx = state.current_node_index()                        # [B]
    cur_dists = dm[torch.arange(dm.size(0)), cur_idx]           # [B,N]
    unvisited_dists = cur_dists * unvisited                     # [B,N]
    # nearest unvisited
    unvisited_dists = torch.where(unvisited, unvisited_dists, torch.inf)
    nearest_unvisited_dist, _ = unvisited_dists.min(dim=1)      # [B]
    start_idx = state.first_node_index()                        # [B]
    back_to_start = dm[torch.arange(dm.size(0)), cur_idx, start_idx]  # [B]
    # combine
    estimate = avg_unvisited_cost + nearest_unvisited_dist + back_to_start
    # scale to keep reasonable magnitude
    return (estimate * 0.1).unsqueeze(1)                        # [B,1]