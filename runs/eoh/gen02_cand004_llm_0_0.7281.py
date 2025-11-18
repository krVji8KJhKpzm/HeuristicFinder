# score=0.728072
# gamma=0.100000
# code_hash=0cc2a43e13841f036b76914d86123d54dbc39c9e3e58d8eeb12c85a3cec64035
# stats: mse=0.408329; rmse=0.639006; mse_tsp100=1.37349; mse_tsp20=0.408329; mse_tsp50=0.737888; mse_worst=1.37349; rmse_tsp100=1.17196; rmse_tsp20=0.639006; rmse_tsp50=0.859004; rmse_worst=1.17196
# ALGORITHM: {auto} def phi(state): """ {Estimate future tour length by summing the minimum connection cost for each unvisited node and adding the cost to return to the start.}
# THOUGHT: {auto}
def phi(state):
    """
    {Estimate future tour length by summing the minimum connection cost for each unvisited node and adding the cost to return to the start.}
    Args:
        state: A view of the current state of the TSP environment.
    Returns:
        value: A tensor of shape [B, 1] representing the estimated future tour length.
    """
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    # [B]
    current_node_idx = state.current_node_index()
    # [B]
    first_node_idx = state.first_node_index()
    # [B]
    batch_indices = torch.arange(dist_matrix.size(0), device=dist_matrix.device)

    # Cost to return to the start node from the current node
    # [B]
    dist_to_start = dist_matrix[batch_indices, current_node_idx, first_node_idx]

    # For each unvisited node, find its minimum connection cost to any other node.
    # We set diagonal to infinity to avoid picking the zero-cost self-connection.
    # [B, N, N]
    dist_matrix_no_self = dist_matrix + torch.diag(torch.full((dist_matrix.size(1),), float('inf'), device=dist_matrix.device))
    # [B, N]
    min_dist_per_node, _ = torch.min(dist_matrix_no_self, dim=2)
    
    # Sum these minimum distances only for the unvisited nodes
    # [B, N]
    min_dist_per_node.masked_fill_(~unvisited_mask, 0)
    # [B]
    sum_min_dists = torch.sum(min_dist_per_node, dim=1)
    
    # The total estimated cost is the sum of minimum connection costs for unvisited nodes
    # plus the cost to return to the start node from the current node.
    # [B]
    value = sum_min_dists + dist_to_start

    return value.unsqueeze(-1)