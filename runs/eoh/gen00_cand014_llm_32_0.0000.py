# score=0.000000
# gamma=-0.100000
# code_hash=0ce4ba3937139f8ee01f1be0ad36944ebf755767eb5c8084fb39248672b83fba
# THOUGHT: {auto}
# THOUGHT: {auto}
def phi(state):
    # Coordinates and masks
    coords = state.all_node_coords()  # [B,N,2]
    unvisited = state.unvisited_mask()  # [B,N]
    current_idx = state.current_node_index()  # [B]
    B, N, _ = coords.shape

    # Current location
    cur_loc = coords[torch.arange(B), current_idx, :]  # [B,2]

    # Distance to all nodes from current
    cur_to_all = torch.linalg.norm(coords - cur_loc.unsqueeze(1), dim=2)  # [B,N]

    # Nearest unvisited distance
    cur_to_unvisited = torch.where(unvisited, cur_to_all, torch.inf)
    nearest_dist = torch.min(cur_to_unvisited, dim=1).values  # [B]

    # Pairwise distances among unvisited nodes
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B,N,N,2]
    dist_mat = torch.linalg.norm(diff, dim=3)  # [B,N,N]

    # Mask to unvisited pairs
    unvisited_pair = unvisited.unsqueeze(1) & unvisited.unsqueeze(2)  # [B,N,N]
    unvisited_pair = unvisited_pair & ~torch.eye(N, dtype=torch.bool, device=coords.device).unsqueeze(0)

    # Mean distance among unvisiteds (MST proxy)
    unvisited_dists = torch.where(unvisited_pair, dist_mat, torch.nan)
    mean_unvisited_dist = torch.nanmean(unvisited_dists, dim=(1, 2))  # [B]

    # Estimate: nearest + (num_unvisited - 1) * mean_unvisited_dist
    num_unvisited = unvisited.sum(dim=1).float()  # [B]
    estimate = nearest_dist + torch.maximum(num_unvisited - 1, torch.tensor(0.0)) * mean_unvisited_dist  # [B]

    # Normalize by average pairwise distance in instance for scale invariance
    avg_instance_dist = dist_mat.mean(dim=(1, 2))  # [B]
    avg_instance_dist = torch.where(avg_instance_dist > 0, avg_instance_dist, torch.tensor(1.0))
    normalized = estimate / avg_instance_dist  # [B]

    # Return as [B,1]
    return normalized.unsqueeze(1)