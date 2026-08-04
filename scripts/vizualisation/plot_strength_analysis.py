"""
plot_strength_analysis.py

Generates a bar/line plot showing test accuracy across noise-strength
sampling ranges, for all three classifiers, at a fixed qubit count (n=3).

Usage:
    python3 plot_strength_analysis.py --output strength_analysis_plot.png
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np


# Data: mean test accuracy across 3 seeds (42, 43, 44), n_qubits=3
STRENGTH_RANGES = ["[0.05, 0.25]", "[0.1, 0.5]", "[0.1, 1.0]"]

RESULTS = {
    "Random Forest": [0.6783, 0.7355, 0.6508],
    "Extra Trees":   [0.6903, 0.7358, 0.6545],
    "Multilayer Perceptron": [0.6737, 0.7150, 0.6248],
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
        default="strength_analysis_plot.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    x = np.arange(len(STRENGTH_RANGES))

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
    ax.set_xticklabels(STRENGTH_RANGES, fontsize=10)
    ax.set_xlabel("Noise Strength Range", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Classification Accuracy vs. Noise Strength Range", fontsize=14)
    ax.set_ylim(60, 80)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=12, loc="upper right")

    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()