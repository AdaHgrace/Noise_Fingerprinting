"""
train_classifier.py

Train classifiers on generated quantum noise fingerprint datasets.

Input:
    data/*.npz containing:
        X      : features
        y      : integer labels
        labels : class names
        meta   : optional metadata

Outputs:
    results/<run_name>/
        metrics.txt
        confusion_matrix.npy
        predictions.npz
        random_forest.joblib
        mlp.joblib

Usage (run from the project root):
    python3 -m scripts.train_classifier --dataset data/your_dataset.npz --models extra_trees,random_forest,mlp
"""

import os
import argparse
import json
from datetime import datetime

import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier


def load_dataset(path):
    """
    Load a noise fingerprint dataset from a .npz file.

    Args:
        path: Path to the .npz file, expected to contain X, y, labels,
            and optionally meta.

    Returns:
        Tuple of (X, y, labels, meta).
    """
    data = np.load(path, allow_pickle=True)

    X = data["X"]
    y = data["y"]
    labels = data["labels"]

    meta = data["meta"] if "meta" in data.files else None

    print("=" * 80)
    print("Loaded dataset")
    print("=" * 80)
    print(f"Path      : {path}")
    print(f"X shape   : {X.shape}")
    print(f"y shape   : {y.shape}")
    print(f"Labels    : {labels.tolist()}")

    unique, counts = np.unique(y, return_counts=True)
    print("\nClass counts:")
    for class_id, count in zip(unique, counts):
        print(f"  {class_id} ({labels[class_id]}): {count}")

    print("=" * 80)

    return X, y, labels, meta


def build_model(model_name, seed):
    """
    Construct a classifier with fixed hyperparameters.

    Hyperparameters are left at sensible defaults rather than tuned,
    so that results reflect the discriminability of the feature
    representation rather than the effect of hyperparameter search.

    Args:
        model_name: One of "random_forest", "extra_trees", "mlp".
        seed: Random seed for the model's internal random_state.

    Returns:
        An unfit scikit-learn estimator (or Pipeline, for "mlp").

    Raises:
        ValueError: If model_name is not recognized.
    """
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=64,
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=seed,
                verbose=True,
            )),
        ])

    raise ValueError(
        f"Unknown model_name={model_name}. "
        "Choose from: random_forest, extra_trees, mlp"
    )


def save_metrics(
    output_dir,
    model_name,
    dataset_path,
    labels,
    y_train,
    y_val,
    y_test,
    y_pred,
    acc,
    f1,
    report,
    cm,
    args,
):
    """
    Save per-model metrics, confusion matrix, and run arguments to disk.
    """
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, f"{model_name}_confusion_matrix.npy"), cm)

    metrics_path = os.path.join(output_dir, f"{model_name}_metrics.txt")

    with open(metrics_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Train size: {len(y_train)}\n")
        f.write(f"Val size  : {len(y_val)}\n")
        f.write(f"Test size : {len(y_test)}\n\n")

        f.write(f"Test accuracy: {acc:.6f}\n")
        f.write(f"Test macro F1: {f1:.6f}\n\n")

        f.write("Labels:\n")
        for i, label in enumerate(labels):
            f.write(f"  {i}: {label}\n")

        f.write("\nClassification report:\n")
        f.write(report)
        f.write("\n\nConfusion matrix:\n")
        f.write(str(cm))
        f.write("\n\nArgs:\n")
        f.write(json.dumps(vars(args), indent=2))

    print(f"Saved metrics to: {metrics_path}")


def train_one_model(
    model_name, X_train, y_train, X_val, y_val, X_test, y_test,
    labels, args, output_dir,
):
    """
    Train one classifier, evaluate it on the validation and test sets,
    and save the model, predictions, and metrics to output_dir.

    Returns:
        Tuple of (test accuracy, test macro F1).
    """
    print("\n" + "=" * 80)
    print(f"Training model: {model_name}")
    print("=" * 80)

    model = build_model(model_name, args.seed)

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="macro")

    report = classification_report(
        y_test,
        y_pred,
        target_names=[str(label) for label in labels],
        digits=4,
    )

    cm = confusion_matrix(y_test, y_pred)

    print(f"\nValidation accuracy: {val_acc:.4f}")
    print(f"Test accuracy      : {test_acc:.4f}")
    print(f"Test macro F1      : {test_f1:.4f}")

    print("\nClassification report:")
    print(report)

    print("\nConfusion matrix:")
    print(cm)

    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"\nSaved model to: {model_path}")

    np.savez_compressed(
        os.path.join(output_dir, f"{model_name}_predictions.npz"),
        y_test=y_test,
        y_pred=y_pred,
        labels=labels,
    )

    save_metrics(
        output_dir=output_dir,
        model_name=model_name,
        dataset_path=args.dataset,
        labels=labels,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        y_pred=y_pred,
        acc=test_acc,
        f1=test_f1,
        report=report,
        cm=cm,
        args=args,
    )

    return test_acc, test_f1


def main():
    parser = argparse.ArgumentParser(
        description="Train classifier on quantum noise fingerprint dataset."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset .npz file.",
    )

    parser.add_argument(
        "--models",
        type=str,
        default="extra_trees,random_forest,mlp",
        help="Comma-separated models: random_forest, extra_trees, mlp",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. If not provided, creates timestamped result folder.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Test split fraction.",
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Validation split fraction from remaining train data.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    X, y, labels, meta = load_dataset(args.dataset)

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
        output_dir = os.path.join("results", f"{dataset_name}_{timestamp}")
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # First split off the test set, then split the remainder into
    # train/validation. The combined effect of test_size=0.20 and
    # val_size=0.15 (applied to the remaining 80%) yields an
    # effective 68/12/20 train/validation/test ratio.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y_trainval,
    )

    print("\nSplit sizes:")
    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Val  : {X_val.shape}, {y_val.shape}")
    print(f"  Test : {X_test.shape}, {y_test.shape}")

    selected_models = [
        item.strip()
        for item in args.models.split(",")
        if item.strip()
    ]

    results = {}

    for model_name in selected_models:
        acc, f1 = train_one_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            labels=labels,
            args=args,
            output_dir=output_dir,
        )

        results[model_name] = {"accuracy": acc, "macro_f1": f1}

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    for model_name, metrics in results.items():
        print(
            f"{model_name:15s}: "
            f"accuracy={metrics['accuracy']:.4f}  "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()