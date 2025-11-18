# score=0.012371
# gamma=0.100000
# code_hash=c07ac942e284504a00f1c059c1e12455fbe237ea11bed1e024492af1fe9e495e
# stats: mse=5.87829; rmse=2.42452; mse_tsp100=80.8318; mse_tsp20=5.87829; mse_tsp50=26.8202; mse_worst=80.8318; rmse_tsp100=8.99065; rmse_tsp20=2.42452; rmse_tsp50=5.17882; rmse_worst=8.99065
# ALGORITHM: {auto}
# THOUGHT: {auto}
def phi(state):
    """
    Estimates the future tour length (cost-to-go) for the TSP.
    It's the sum of two components:
    1. A cost related to the remaining sub-tour of unvisited nodes.
    2. A cost for connecting the current partial tour to the remaining sub-tour.
    """
    # [B, N, N]
    dist_matrix = state.distance_matrix()
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    # [B]
    num_unvisited = unvisited_mask.float().sum(dim=1)

    # Handle terminal state: if no nodes are unvisited, the future cost is zero.
    is_terminal = (num_unvisited == 0)

    # 1. Estimate the cost of the sub-tour connecting the unvisited nodes.
    # For each unvisited node, find the minimum distance to another unvisited node.
    # [B, N, N], set distances to/from visited nodes to infinity
    unvisited_dist = dist_matrix.clone()
    unvisited_dist_mask = ~(unvisited_mask.unsqueeze(2) & unvisited_mask.unsqueeze(1))
    unvisited_dist[unvisited_dist_mask] = float('inf')

    # Set diagonal to infinity to find min distance to *another* unvisited node
    unvisited_dist.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))

    # [B, N], min distance from each unvisited node to any other unvisited node
    min_dist_per_node, _ = torch.min(unvisited_dist, dim=2)
    # Zero out distances for visited nodes
    min_dist_per_node[~unvisited_mask] = 0.0
    # [B], sum of these minimum distances
    subtour_cost = min_dist_per_node.sum(dim=1)

    # 2. Estimate the cost to connect the current path to the unvisited sub-tour.
    # Find the distance from the current node to the nearest unvisited node.
    # [B]
    current_node = state.current_node_index()
    # [B, N], distances from the current node
    dist_from_current = torch.gather(dist_matrix, 1, current_node.view(-1, 1, 1).expand(-1, 1, dist_matrix.size(2))).squeeze(1)

    # [B, N], set distances to visited nodes to infinity
    dist_from_current_to_unvisited = dist_from_current.clone()
    dist_from_current_to_unvisited[~unvisited_mask] = float('inf')
    # [B], min distance to an unvisited node
    connection_cost, _ = torch.min(dist_from_current_to_unvisited, dim=1)
    # If terminal, connection_cost is inf; set to 0.
    connection_cost = torch.nan_to_num(connection_cost, posinf=0.0)

    # Scaling factor to account for the increasing complexity/length with more nodes
    # Use log1p to handle num_unvisited=0 gracefully (log1p(0)=0) and smooth the scaling
    scaling_factor = torch.log1p(num_unvisited)

    # Combine the components
    # The heuristic is a scaled sum of the sub-tour cost and the connection cost.
    value = (subtour_cost + connection_cost) * scaling_factor

    # Ensure terminal state has a value of 0
    value[is_terminal] = 0.0

    # The value represents a cost-to-go, so it should be negative for reward shaping.
    # We return a negative value as V(s) is typically negated in shaping: r' = r + gamma*V(s') - V(s).
    # Since tour length is a cost (negative reward), V(s) should be positive.
    return value.unsqueeze(-1)