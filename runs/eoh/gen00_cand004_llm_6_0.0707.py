# score=0.070741
# gamma=0.100000
# code_hash=6237d31dbca11f5a5ae019268f650c373ab5374f87f40f348a0668cf45d19af3
# stats: mse=14.136; rmse=3.75979; mse_tsp100=577.304; mse_tsp20=14.136; mse_tsp50=128.437; rmse_tsp100=24.0271; rmse_tsp20=3.75979; rmse_tsp50=11.333
# ALGORITHM: {auto} def phi(state): """ Estimates the future tour length (cost-to-go) for the TSP environment. {Algorithm: Calculate the expected cost by summing two components: (1) the average distance from the current node to all unvisited nodes, representing the next step's cost, and (2) the expected cost to traverse the remaining unvisited nodes, approximated by multiplying the number of remaining steps (num_unvisited - 1) by the average pairwise distance among all unvisited nodes.}
# THOUGHT: {auto}
def phi(state):
    """
    Estimates the future tour length (cost-to-go) for the TSP environment.
    {Algorithm: Calculate the expected cost by summing two components: (1) the average distance from the current node to all unvisited nodes, representing the next step's cost, and (2) the expected cost to traverse the remaining unvisited nodes, approximated by multiplying the number of remaining steps (num_unvisited - 1) by the average pairwise distance among all unvisited nodes.}
    Args:
        state: A TSPStateView object with batch-friendly helper methods.
    Returns:
        A torch tensor of shape [B, 1] representing the estimated future tour length.
    """
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N]
    unvisited = state.unvisited_mask()
    # [B]
    current_node = state.current_node_index()
    # [B]
    num_unvisited = unvisited.float().sum(dim=1)
    # A small epsilon to prevent division by zero when num_unvisited is 0 or 1
    epsilon = 1e-9

    # --- Component 1: Expected cost of the next step ---
    # Gather distances from the current node: [B, N]
    current_node_dists = dist_matrix.gather(1, current_node.view(-1, 1, 1).expand(-1, 1, dist_matrix.size(2))).squeeze(1)
    # Mask to keep only distances to unvisited nodes
    current_to_unvisited_dists = current_node_dists * unvisited.float()
    # Average distance from current to unvisited nodes: [B]
    avg_dist_to_unvisited = current_to_unvisited_dists.sum(dim=1) / (num_unvisited + epsilon)

    # --- Component 2: Expected cost of the remaining tour among unvisited nodes ---
    # Create a mask for unvisited-to-unvisited pairs: [B, N, N]
    unvisited_mask_2d = unvisited.unsqueeze(2) & unvisited.unsqueeze(1)
    # Sum of distances between all pairs of unvisited nodes: [B]
    total_unvisited_dist = (dist_matrix * unvisited_mask_2d.float()).sum(dim=[1, 2])
    # Number of pairs of unvisited nodes (N * (N-1)): [B]
    num_pairs = num_unvisited * (num_unvisited - 1)
    # Average pairwise distance among unvisited nodes: [B]
    avg_pairwise_dist = total_unvisited_dist / (num_pairs + epsilon)

    # The number of edges remaining in the sub-tour of unvisited nodes is (num_unvisited - 1)
    # This is because we need to connect all unvisited nodes, and then one more edge to connect back to start (handled by the environment reward)
    # We approximate the sub-tour length with (num_unvisited - 1) * avg_pairwise_dist
    remaining_steps = torch.clamp(num_unvisited - 1, min=0)
    remaining_tour_cost = remaining_steps * avg_pairwise_dist

    # --- Total Estimated Future Cost ---
    # The total cost is the cost to take the next step plus the cost of the remaining sub-tour.
    # We use a mask to set the value to 0 for terminal states (num_unvisited == 0).
    is_not_done = (num_unvisited > 0).float()
    value = (avg_dist_to_unvisited + remaining_tour_cost) * is_not_done

    # Ensure the output shape is [B, 1]
    return value.unsqueeze(1)