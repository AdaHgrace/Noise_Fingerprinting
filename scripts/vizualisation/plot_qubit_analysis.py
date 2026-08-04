"""
plot_qubit_analysis.py

Generates a plot showing test accuracy across qubit counts (2, 3, 4),
for all three classifiers, at a fixed strength range ([0.1, 0.5]).

Usage:
    python3 plot_qubit_analysis.py --output qubit_analysis_plot.png
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np


# Data: mean test accuracy across 3 seeds (42, 43, 44), strength range [0.1, 0.5]
QUBIT_COUNTS = [2, 3, 4]

RESULTS = {
    "Random Forest": [0.7527, 0.7355, 0.7445],
    "Extra Trees":   [0.7467, 0.7358, 0.7340],
    "Multilayer Perceptron": [0.7327, 0.7150, 0.7140],
}

COLORS = {
    "Random Forest": "blue",
    "Extra Trees": "red",
    "Multilayer Perceptron": "green",
}

MARKERS = {
    "Random Forest": "o",
    "Extra Trees": "s",
    "Multilayer Perceptron": "^",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="qubit_analysis_plot.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    x = np.array(QUBIT_COUNTS)

    fig, ax = plt.subplots(figsize=(9, 4))

    for model_name, accuracies in RESULTS.items():
        ax.plot(
            x,
            [a * 100 for a in accuracies],
            marker=MARKERS[model_name],
            color=COLORS[model_name],
            label=model_name,
            linewidth=2,
            markersize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(q) for q in QUBIT_COUNTS], fontsize=10)
    ax.set_xlabel("Number of Qubits", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Classification Accuracy vs. Qubit Count", fontsize=14)
    ax.set_ylim(68, 78)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=12, loc="upper right")

    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()