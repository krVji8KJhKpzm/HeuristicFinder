# score=0.000000
# gamma=0.100000
# code_hash=9d3486f7872add595bd26b523f9480f221bfc986e56b1a49f7263e3df4953036
# THOUGHT: {auto}
# THOUGHT: {auto}
def phi(state):
    coords = state.all_node_coords()  # [B,N,2]
    B, N, _ = coords.shape
    unvisited = state.unvisited_mask()  # [B,N]
    num_left = unvisited.sum(dim=1, keepdim=True).clamp(min=1)  # [B,1]
    frac_left = num_left / N  # [B,1]

    # pairwise distances among unvisited
    dist_mat = state.distance_matrix()  # [B,N,N]
    unvisited_f = unvisited.float()
    mask2d = unvisited.unsqueeze(1) * unvisited.unsqueeze(2)  # [B,N,N]
    masked_d = dist_mat * mask2d.float()
    avg_pair = (masked_d.sum(dim=(1,2)) / (num_left.squeeze(1) * (num_left.squeeze(1) - 1)).clamp(min=1)).unsqueeze(1)  # [B,1]

    # distances from current node to unvisited
    cur_idx = state.current_node_index()  # [B]
    cur_d = dist_mat[torch.arange(B), cur_idx]  # [B,N]
    cur_d = cur_d * unvisited_f
    nearest_d_cur, nearest_idx = cur_d.min(dim=1)  # [B]

    # distance from nearest unvisited to first node
    first_idx = state.first_node_index()  # [B]
    back_d = dist_mat[torch.arange(B), nearest_idx, first_idx]  # [B]

    # heuristic: (avg pairwise) + (cur->nearest) + (nearest->start)
    heuristic = avg_pair + nearest_d_cur.unsqueeze(1) + back_d.unsqueeze(1)
    # scale by fraction left to ensure terminal consistency
    return heuristic * frac_left  # [B,1]