# score=0.062435
# gamma=1.000000
# code_hash=627d91b23b0370ff838ed2c69bc931795333336d40829148afcc16965d6b24a6
# stats: mse=3.28463; rmse=1.81236; mse_tsp100=16.0168; mse_tsp20=3.28463; mse_tsp50=8.05051; mse_worst=16.0168; rmse_tsp100=4.0021; rmse_tsp20=1.81236; rmse_tsp50=2.83734; rmse_worst=4.0021
# ALGORITHM: {auto} def phi(state): """ {Estimates future tour length based on the convex hull area of unvisited nodes, plus connection costs.}
# THOUGHT: {auto}
def phi(state):
    """
    {Estimates future tour length based on the convex hull area of unvisited nodes, plus connection costs.}
    The heuristic has three parts:
    1. The area of the convex hull of the unvisited nodes, scaled by a factor. This approximates the density and spread of the remaining problem.
    2. The cost to connect the current node to the closest unvisited node.
    3. The cost to connect the start node to the closest unvisited node (approximating the tour completion cost).
    """
    # [B, N, 2]
    coords = state.all_node_coords()
    # [B, N]
    unvisited_mask = state.unvisited_mask()
    # [B]
    num_unvisited = unvisited_mask.sum(dim=1)
    # [B, N, N]
    dist_matrix = state.distance_matrix()

    # Handle terminal state: if no nodes are unvisited, the future cost is zero.
    is_terminal = (num_unvisited == 0)
    # Handle case with < 3 unvisited nodes where convex hull is undefined/zero area
    is_small_subproblem = (num_unvisited < 3)

    # 1. Convex Hull Area of Unvisited Nodes
    # Set coordinates of visited nodes to a large value so they don't affect the hull
    masked_coords = coords.clone()
    masked_coords[~unvisited_mask] = 1e9
    
    # Find the node with the minimum y-coordinate (and then x) to start the hull scan
    min_y_coords, _ = masked_coords[:, :, 1].min(dim=1, keepdim=True)
    is_min_y = (masked_coords[:, :, 1] == min_y_coords)
    min_x_at_min_y, _ = (masked_coords[:, :, 0] + (~is_min_y) * 1e9).min(dim=1, keepdim=True)
    is_start_node = is_min_y & (masked_coords[:, :, 0] == min_x_at_min_y)
    start_node_idx = torch.argmax(is_start_node.float(), dim=1)
    
    # Calculate angles from the start node to all other unvisited nodes
    start_node_coords = torch.gather(coords, 1, start_node_idx.view(-1, 1, 1).expand(-1, 1, 2))
    delta = masked_coords - start_node_coords
    angles = torch.atan2(delta[:, :, 1], delta[:, :, 0])
    angles[~unvisited_mask] = 4.0 # Put visited nodes at the end after sorting
    angles.scatter_(1, start_node_idx.unsqueeze(1), 4.0) # Start node also at the end

    # Sort nodes by angle to get the convex hull vertex order
    _, sorted_indices = torch.sort(angles, dim=1)
    sorted_coords = torch.gather(coords, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, 2))

    # Shoelace formula for polygon area
    x = sorted_coords[:, :, 0]
    y = sorted_coords[:, :, 1]
    # Area = 0.5 * |(x1*y2 + x2*y3 + ... + xn*y1) - (y1*x2 + y2*x3 + ... + yn*x1)|
    area = 0.5 * torch.abs(torch.sum(x * torch.roll(y, -1, dims=1), dim=1) - torch.sum(y * torch.roll(x, -1, dims=1), dim=1))
    
    # The area is an approximation of the subproblem size. Scale it to be distance-like.
    # Sqrt(area) is a length-like dimension.
    hull_cost = torch.sqrt(area)
    hull_cost[is_small_subproblem] = 0.0 # No area for <3 points

    # 2. Connection Costs
    current_node = state.current_node_index()
    start_node = state.first_node_index()

    # Distances from current/start to all nodes
    dist_from_current = torch.gather(dist_matrix, 1, current_node.view(-1, 1, 1).expand(-1, 1, dist_matrix.size(2))).squeeze(1)
    dist_from_start = torch.gather(dist_matrix, 1, start_node.view(-1, 1, 1).expand(-1, 1, dist_matrix.size(2))).squeeze(1)

    # Mask to find closest unvisited node
    dist_to_unvisited = dist_from_current.clone()
    dist_to_unvisited[~unvisited_mask] = float('inf')
    connect_cost_current, _ = torch.min(dist_to_unvisited, dim=1, keepdim=False)

    dist_to_unvisited_from_start = dist_from_start.clone()
    dist_to_unvisited_from_start[~unvisited_mask] = float('inf')
    connect_cost_start, _ = torch.min(dist_to_unvisited_from_start, dim=1, keepdim=False)

    # Combine costs
    value = hull_cost + connect_cost_current + connect_cost_start
    value[is_terminal] = 0.0
    
    return value.unsqueeze(-1)