# score=0.000000
# gamma=-0.100000
# code_hash=ff71331ea540fe964c419de5dd8018dd2948f416846d197503cdb090b3520ffc
# THOUGHT: {auto}
# THOUGHT: {auto}
def phi(state):
    B = state.current_node_index().size(0)
    unvisited = state.unvisited_mask()  # [B, N]
    N = unvisited.size(1)
    dist = state.distance_matrix()  # [B, N, N]
    cur_idx = state.current_node_index()  # [B]
    # distances from current node to all nodes
    cur_dists = dist[torch.arange(B), cur_idx]  # [B, N]
    # mask out visited (set to inf to ignore in min)
    cur_dists = torch.where(unvisited, cur_dists, torch.tensor(float('inf'), device=cur_dists.device))
    # nearest unvisited from current
    nearest_cur = torch.min(cur_dists, dim=1)[0]  # [B]
    # pairwise distances among unvisited nodes
    # create a mask [B, N, N] for unvisited pairs
    unvisited_bcast = unvisited.unsqueeze(1) & unvisited.unsqueeze(2)  # [B, N, N]
    # set non-unvisited pairs to inf
    unvisited_dists = torch.where(unvisited_bcast, dist, torch.tensor(float('inf'), device=dist.device))
    # min outgoing edge per unvisited node (over unvisited only)
    min_out = torch.min(unvisited_dists, dim=2)[0]  # [B, N]
    # mask out visited (set to 0 for reduction)
    min_out = torch.where(unvisited, min_out, torch.tensor(0.0, device=min_out.device))
    # count unvisited
    num_unvisited = unvisited.sum(dim=1).float()  # [B]
    # MST-like lower bound: sum of min outgoing edges divided by 2 (1/2 * sum(min_edges) is a lower bound on MST)
    mst_bound = 0.5 * min_out.sum(dim=1)  # [B]
    # fallback when no unvisited
    mst_bound = torch.where(num_unvisited > 0, mst_bound, torch.tensor(0.0, device=mst_bound.device))
    # heuristic: nearest from current + mst_bound over unvisited
    est = nearest_cur + mst_bound
    # scale to reasonable magnitude (empirically ~ N/2 steps, each ~unit distance)
    est = est * (num_unvisited + 1).sqrt()
    # ensure [B, 1] output
    return est.unsqueeze(1)