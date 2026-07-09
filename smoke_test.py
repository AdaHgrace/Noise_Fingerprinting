"""
smoke_test.py

End-to-end smoke test for the Shadow-Based Noise Fingerprinting pipeline.

This script verifies that the full pipeline runs correctly by generating
a small dataset and training a classifier on it using the same scripts
used to produce the paper results.

WARNING
-------
This smoke test uses a very small dataset (50 samples per class, 500
total) to complete quickly. Classification accuracy on this dataset will
be substantially lower than the results reported in the paper (~84%),
which were obtained with 1,400 samples per class. Low accuracy here
(typically 20-50%) is expected and does not indicate a problem with
your installation.

To reproduce the paper results, use the full dataset generation command
documented in the README.

Usage:
    python3 smoke_test.py

Expected runtime: 5-15 minutes depending on hardware.
"""

import os
import sys
import json
import tempfile
import numpy as np


DATASET_PATH = os.path.join(tempfile.gettempdir(), "smoke_test_dataset.npz")
RESULTS_DIR = os.path.join(tempfile.gettempdir(), "smoke_test_results")


def check(condition, message):
    if not condition:
        print(f"FAILED: {message}")
        sys.exit(1)


def run_smoke_test():
    print("=" * 70)
    print("Shadow-Based Noise Fingerprinting — Smoke Test")
    print("=" * 70)
    print()
    print("WARNING: This test uses only 50 samples per class (500 total).")
    print("Accuracy will be much lower than the paper results (~84%).")
    print("Low accuracy here is expected and does not indicate a problem")
    print("with your installation.")
    print()

    # ------------------------------------------------------------------ #
    # Step 1: Generate a small dataset
    # ------------------------------------------------------------------ #
    print("[1/3] Generating small dataset (50 samples/class, 10 noise types)...")
    print("      This may take several minutes...")
    print()

    ret = os.system(
        f"python3 -m scripts.generate_dataset "
        f"--output {DATASET_PATH} "
        f"--samples-per-class 50 "
        f"--shots 50 "
        f"--n-qubits 3 "
        f"--num-qaoa-probes 5 "
        f"--num-workers 8 "
        f"--noise-types all"
    )

    check(ret == 0, "Dataset generation script failed.")
    check(os.path.exists(DATASET_PATH), "Dataset file was not created.")

    # Verify dataset contents
    data = np.load(DATASET_PATH, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    labels = data["labels"]

    check(X.shape[0] == 500, f"Expected 500 samples, got {X.shape[0]}")
    check(X.shape[1] == 279, f"Expected 279 features, got {X.shape[1]}")
    check(len(np.unique(y)) == 10, f"Expected 10 classes, got {len(np.unique(y))}")
    check(not np.isnan(X).any(), "Dataset contains NaN values")
    check(not np.isinf(X).any(), "Dataset contains Inf values")

    print(f"      X shape : {X.shape}")
    print(f"      Labels  : {labels.tolist()}")
    print("      Dataset checks passed.")
    print()
    

    # ------------------------------------------------------------------ #
    # Step 2: Train classifier using the actual training script
    # ------------------------------------------------------------------ #
    print("[2/3] Training Random Forest classifier...")
    print("      Using scripts/train_classifier.py ...")
    print()

    ret = os.system(
        f"python3 -m scripts.train_classifier "
        f"--dataset {DATASET_PATH} "
        f"--models random_forest "
        f"--output-dir {RESULTS_DIR} "
        f"--seed 42"
    )

    check(ret == 0, "Training script failed.")

    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    check(os.path.exists(summary_path), "summary.json was not created.")

    with open(summary_path) as f:
        summary = json.load(f)

    acc = summary["random_forest"]["accuracy"] * 100
    f1 = summary["random_forest"]["macro_f1"]

    print()
    print(f"      Test accuracy : {acc:.1f}%")
    print(f"      Macro F1      : {f1:.4f}")
    print()

    # ------------------------------------------------------------------ #
    # Step 3: Verify outputs exist
    # ------------------------------------------------------------------ #
    print("[3/3] Verifying output files...")

    expected_files = [
        "summary.json",
        "random_forest.joblib",
        "random_forest_metrics.txt",
        "random_forest_confusion_matrix.npy",
        "random_forest_predictions.npz",
    ]

    for fname in expected_files:
        fpath = os.path.join(RESULTS_DIR, fname)
        check(os.path.exists(fpath), f"Expected output file not found: {fname}")
        print(f"      Found: {fname}")

    print()
    print("=" * 70)
    print("Smoke test PASSED — pipeline is working correctly.")
    print()
    print("NOTE: Accuracy on this small dataset is expected to be low.")
    print(f"      Got {acc:.1f}% — paper reports 84.26% with 1,400 samples/class.")
    print("      See README for full dataset generation instructions.")
    print("=" * 70)
    
    # Cleanup
    os.remove(DATASET_PATH)
    import shutil
    shutil.rmtree(RESULTS_DIR, ignore_errors=True)


if __name__ == "__main__":
    run_smoke_test()