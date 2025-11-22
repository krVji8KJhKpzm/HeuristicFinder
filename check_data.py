import torch

path = "data/tsp100_listwise_offline_trajs_100000.pt"
data = torch.load(path, map_location="cpu")
episodes = data["episodes"]

final_rewards = torch.tensor(
    [ep["final_reward"] for ep in episodes], dtype=torch.float32
)
avg_tour_len = (-final_rewards).mean().item()   # final_reward 是负的

print("平均 tour 长度:", avg_tour_len)
