def phi(state):
    # Use ONLY raw helpers: action/visited masks, current/first indices, distance_matrix
    dmat = state.distance_matrix(normalize=True)  # [B,N,N]
    mask = state.action_mask()  # [B,N] (True = unvisited)
    cur = state.current_node_index().long()  # [B]

    B = dmat.shape[0]
    idx = torch.arange(B, device=dmat.device)
    dc = dmat[idx, cur]  # distances from current node to all nodes [B,N]

    unv = mask
    cnt = unv.sum(dim=-1, keepdim=True).clamp_min(1)

    # Mean distance to unvisited nodes
    mean_unv = (dc * unv.float()).sum(dim=-1, keepdim=True) / cnt

    # Nearest unvisited distance (robust when none remain)
    big = torch.finfo(dc.dtype).max
    nearest = torch.min(torch.where(unv, dc, torch.full_like(dc, big)), dim=-1, keepdim=True).values
    nearest = torch.where(torch.isinf(nearest), torch.zeros_like(nearest), nearest)

    value = -(0.7 * nearest + 0.3 * mean_unv)
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    value = torch.clamp(value, min=-10.0, max=10.0)
    return value
