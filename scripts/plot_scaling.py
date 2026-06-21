"""
plot_scaling.py

Plot test accuracy as a function of dataset size(samples per class)
for Random Forest, Extra Trees, and MLP classifiers.
"""

import json
import os

import matplotlib.pyplot as plt


# Each entry maps a samples-per-class value to the results directory
# produced by train_classifier.py for that dataset size.
RUNS = {
    300: "results/qaoa_dataset10_20260519_071922",
    700: "results/qaoa_dataset12_20260614_121124",
    1050: "results/qaoa_dataset14_20260614_122052",
    1400: "results/qaoa_dataset13_20260525_172053",
}

OUTPUT_DIR = "results/qaoa_dataset13_20260525_172053/visualizations"


def load_accuracies(run_dir):
    """
    Load accuracy for each classifier from a run's summary.json.

    Args:
        run_dir: Path to a results directory containing summary.json.

    Returns:
        Dict mapping model name to test accuracy (as a fraction).
    """
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path) as f:
        data = json.load(f)

    # summary.json entries may be either a plain float (older format)
    # or a dict with "accuracy" and "macro_f1" (current format).
    accuracies = {}
    for model, value in data.items():
        if isinstance(value, dict):
            accuracies[model] = value["accuracy"]
        else:
            accuracies[model] = value

    return accuracies


def main():
    samples_per_class = sorted(RUNS.keys())

    rf_acc, et_acc, mlp_acc = [], [], []

    for n in samples_per_class:
        accuracies = load_accuracies(RUNS[n])
        rf_acc.append(accuracies["random_forest"] * 100)
        et_acc.append(accuracies["extra_trees"] * 100)
        mlp_acc.append(accuracies["mlp"] * 100)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        samples_per_class, rf_acc, marker="o",
        label="Random Forest", linewidth=2, color="blue",
    )
    ax.plot(
        samples_per_class, et_acc, marker="s",
        label="Extra Trees", linewidth=2, color="red",
    )
    ax.plot(
        samples_per_class, mlp_acc, marker="^",
        label="Multilayer Perceptron", linewidth=2, color="green",
    )

    ax.set_xlabel("Samples per Class", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Classification Accuracy vs. Dataset Size", fontsize=13)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(40, 90)
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/scaling_plot.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUTPUT_DIR}/scaling_plot.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()