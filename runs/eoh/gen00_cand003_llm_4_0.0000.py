# score=0.000000
# gamma=1.000000
# code_hash=04783b16e91cec24ee18b14ed1b769c6ba9131d903a136792573a5ed4ecfef5b
# THOUGHT: {auto}
# THOUGHT: {auto}
def phi(state):
    # Fetch data
    coords = state.all_node_coords()  # [B,N,2]
    unvisited = state.unvisited_mask()  # [B,N]
    cur_idx = state.current_node_index()  # [B]
    first_idx = state.first_node_index()  # [B]
    B, N, _ = coords.shape

    # Degenerate case: all visited
    all_done = ~unvisited.any(dim=1, keepdim=True)  # [B,1]

    # 1) Distance from current to nearest unvisited
    dist_to_cur = torch.linalg.norm(
        coords - coords[torch.arange(B), cur_idx].unsqueeze(1), dim=2
    )  # [B,N]
    dist_to_cur = torch.where(unvisited, dist_to_cur, torch.inf)
    cur2next = dist_to_cur.min(dim=1, keepdim=True)[0]  # [B,1]

    # 2) Pairwise distances among unvisited nodes
    u_coords = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B,N,N,2]
    pairwise = torch.linalg.norm(u_coords, dim=3)  # [B,N,N]
    pairwise = torch.where(
        unvisited.unsqueeze(1) & unvisited.unsqueeze(2), pairwise, torch.nan
    )
    avg_pair = pairwise.nanmean(dim=(1, 2), keepdim=True)  # [B,1]

    # 3) Distance from last unvisited back to start
    dist_to_first = torch.linalg.norm(
        coords - coords[torch.arange(B), first_idx].unsqueeze(1), dim=2
    )  # [B,N]
    dist_to_first = torch.where(unvisited, dist_to_first, torch.inf)
    last2first = dist_to_first.min(dim=1, keepdim=True)[0]  # [B,1]

    # Combine: cur2next + (#unvisited-1)*avg_pair + last2first
    n_unvis = unvisited.sum(dim=1, keepdim=True).float()  # [B,1]
    est = cur2next + torch.relu(n_unvis - 1) * avg_pair + last2first
    est = torch.where(all_done, 0.0, est)  # zero when tour complete
    return est  # [B,1]