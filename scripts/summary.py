"""
summary.py

Aggregate accuracy and macro F1 across multiple training runs
(different random seeds) into mean +/- std for each classifier.

Usage:
    python3 -m scripts.summary
"""

import json
import numpy as np

SEEDS = [42, 43, 44]
MODELS = ["random_forest", "extra_trees", "mlp"]


def main():
    results = {model: {"accuracy": [], "macro_f1": []} for model in MODELS}

    for seed in SEEDS:
        with open(f"results/seed{seed}/summary.json") as f:
            data = json.load(f)

        for model in MODELS:
            results[model]["accuracy"].append(data[model]["accuracy"])
            results[model]["macro_f1"].append(data[model]["macro_f1"])

    for model in MODELS:
        acc_mean = np.mean(results[model]["accuracy"]) * 100
        acc_std = np.std(results[model]["accuracy"]) * 100
        f1_mean = np.mean(results[model]["macro_f1"])
        f1_std = np.std(results[model]["macro_f1"])

        print(
            f"{model:15s}: "
            f"accuracy={acc_mean:.2f} +/- {acc_std:.2f}  "
            f"macro_f1={f1_mean:.4f} +/- {f1_std:.4f}"
        )


if __name__ == "__main__":
    main()