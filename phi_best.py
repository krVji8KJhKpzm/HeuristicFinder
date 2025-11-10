def phi(state):
    start_dist = state.distance_to_start(normalize=True)
    centroid_dist = state.distance_to_centroid(normalize=True)
    nearest_dist = state.nearest_unvisited_distance(normalize=True)
    remaining_ratio = state.remaining_ratio()
    step_ratio = state.step_ratio()
    
    potential = 0.4 * start_dist + 0.3 * centroid_dist + 0.3 * nearest_dist
    potential = potential * (remaining_ratio + 0.5 * step_ratio)
    
    return potential