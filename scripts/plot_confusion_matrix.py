"""
plot_confusion_matrix.py

Plot the confusion matrix for a trained classifier.
"""

import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


DATASET_PATH = "data/your_dataset.npz"
CONFUSION_MATRIX_PATH = (
    "results/qaoa_dataset13_20260525_172053/random_forest_confusion_matrix.npy"
)
OUTPUT_DIR = "results/qaoa_dataset13_20260525_172053/visualizations"


def format_label(label):
    """
    Convert a raw label string (e.g. "phase_amplitude_damping") into
    a display-friendly title (e.g. "Phase Amplitude Damping").
    """
    return label.replace("_", " ").title()


def main():
    # Load labels directly from the dataset, in the same order used
    # during training, rather than hardcoding a list that could drift
    # out of sync with the actual label-to-index mapping.
    dataset = np.load(DATASET_PATH, allow_pickle=True)
    raw_labels = dataset["labels"].tolist()
    labels = [format_label(label) for label in raw_labels]

    cm = np.load(CONFUSION_MATRIX_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix - Random Forest Classifier", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()