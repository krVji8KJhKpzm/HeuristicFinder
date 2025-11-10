"""
进化式 Phi(s) 搜索示例：
- 使用 POMOPBRS 在 TSP-20 上短时训练作为适应度评估
- 如本地已配置 Ollama，可通过大模型生成候选；否则回退到本地变异
"""

from rl4co.heuristic_finder.evosearch import EvoConfig, evolution_search


def main():
    cfg = EvoConfig(
        population_size=4,
        survivors=2,
        iterations=2,
        offspring_per_iter=2,
        epochs_per_eval=1,
        batch_size=64,
        train_size=1000,
        val_size=256,
        num_starts=8,
        device="cpu",
        # 如已安装并拉取模型，可启用：
        # ollama_model="qwen3:32b",
    )
    results = evolution_search(cfg)
    print("=== Top Candidates ===")
    for spec, score in results:
        print(f"{spec.name}: val/reward={score:.4f}")


if __name__ == "__main__":
    main()

