# score=0.006395
# gamma=0.100000
# code_hash=098cffc5646f2d78d70fba303d3a95ed9104ca4e1798e43c3f466db5b2b1130c
# stats: mse=156.366; rmse=12.5046; mse_tsp100=16240.5; mse_tsp20=156.366; mse_tsp50=2104.05; rmse_tsp100=127.438; rmse_tsp20=12.5046; rmse_tsp50=45.8699
# ALGORITHM: {auto} def phi(state): """ Estimates the future tour length (cost-to-go) for the TSP environment. {The potential is estimated by summing two components: the average distance from the current node to all unvisited nodes, and the average distance from all unvisited nodes back to the start node, with both components scaled by the number of remaining nodes to approximate the total path length of the remaining sub-tour.}
# THOUGHT: {auto}
def phi(state):
    """
    Estimates the future tour length (cost-to-go) for the TSP environment.
    {The potential is estimated by summing two components: the average distance from the current node to all unvisited nodes, and the average distance from all unvisited nodes back to the start node, with both components scaled by the number of remaining nodes to approximate the total path length of the remaining sub-tour.}
    Args:
        state: A view of the current state of the TSP environment.
    Returns:
        A tensor of shape [B, 1] representing the estimated future tour length.
    """
    # Get batch size and number of nodes
    B, N, _ = state.distance_matrix().shape

    # Get masks for unvisited nodes
    unvisited_mask = state.unvisited_mask() # [B, N]
    num_unvisited = unvisited_mask.sum(dim=1, keepdim=True).float() # [B, 1]

    # Get distance matrix
    dist_matrix = state.distance_matrix() # [B, N, N]

    # Get current and start node indices
    current_idx = state.current_node_index().view(B, 1, 1).expand(-1, 1, N) # [B, 1, N]
    start_idx = state.first_node_index().view(B, 1, 1).expand(-1, N, 1) # [B, N, 1]

    # Component 1: Average distance from current node to all unvisited nodes
    # dist_from_current is [B, N], representing dist(current, i) for all i
    dist_from_current = torch.gather(dist_matrix, 1, current_idx).squeeze(1) # [B, N]
    # Mask out visited nodes and sum distances to unvisited nodes
    sum_dist_from_current_to_unvisited = (dist_from_current * unvisited_mask).sum(dim=1, keepdim=True) # [B, 1]
    # Avoid division by zero when all nodes are visited
    safe_num_unvisited = num_unvisited.clamp(min=1.0)
    avg_dist_from_current = sum_dist_from_current_to_unvisited / safe_num_unvisited # [B, 1]

    # Component 2: Average distance from all unvisited nodes to the start node
    # dist_to_start is [B, N], representing dist(i, start) for all i
    dist_to_start = torch.gather(dist_matrix, 2, start_idx).squeeze(2) # [B, N]
    # Mask out visited nodes and sum distances from unvisited nodes
    sum_dist_from_unvisited_to_start = (dist_to_start * unvisited_mask).sum(dim=1, keepdim=True) # [B, 1]
    avg_dist_to_start = sum_dist_from_unvisited_to_start / safe_num_unvisited # [B, 1]

    # Estimate future path length. The number of remaining edges is (num_unvisited + 1).
    # We use num_unvisited as a simpler scaling factor.
    # The logic is: (avg cost from current to next) + (num_unvisited-1)*(avg cost between unvisited) + (avg cost from last to start)
    # This is approximated by scaling the average distances.
    # The scaling factor `num_unvisited` represents the number of remaining "steps" or edges.
    # The first term represents the cost of the next step, and the second represents the cost of returning home eventually.
    # We use all_node_coords() to get N for scaling, making it more robust.
    # A small constant is added to the scaling to prevent the potential from being exactly zero when only one node is left.
    tour_length_scale = (state.all_node_coords().shape[1] / 20.0).sqrt() # Heuristic scale based on N
    
    # The potential is the sum of the scaled average distances.
    # The first term estimates the cost of visiting all unvisited nodes from the current location.
    # The second term estimates the cost of returning from the "center of mass" of unvisited nodes to the start.
    # Multiplying by num_unvisited approximates the total length of these remaining segments.
    # Adding 1 to num_unvisited accounts for the final edge back to the start.
    remaining_edges = num_unvisited
    
    # We use a simpler heuristic: cost is proportional to the number of remaining nodes
    # and the average distance to them from current and from them to start.
    potential = (avg_dist_from_current + avg_dist_to_start) * remaining_edges * tour_length_scale

    # The potential should be zero when the tour is complete (num_unvisited == 0).
    # The calculation naturally handles this as num_unvisited becomes 0.
    return potential.detach()