"""
generate_dataset.py

Generate a dataset of quantum noise fingerprints using:
- Qiskit Aer noise models
- classical-shadow-based observable estimation
- structured probe circuits
- raw + derived physical features

Output:
    data/<dataset_name>.npz

The saved file contains:
    X      : feature matrix, shape (num_samples, num_features)
    y      : integer labels
    labels : noise type names
    meta   : metadata for each sample

A small number of samples can occasionally fail with rare,
non-deterministic Aer numerical errors (e.g. "Kraus is empty",
Hermitian eigensolver failures on near-degenerate density matrices).
These are now SKIPPED rather than crashing the whole run -- their
details are logged to a separate _failed_tasks.json file so they can
be retried/recovered afterward (see recover_failed_tasks.py).

Usage (run from the project root):
    python3 -m scripts.generate_dataset \
    --output data/my_dataset.npz \
    --samples-per-class 1000 \
    --shots 200 \
    --n-qubits 3 \
    --num-qaoa-probes 5 \
    --num-workers 4 \
    --noise-types all \
    --min-strength 0.1 \
    --max-strength 0.5
"""


import json
import argparse
from pathlib import Path
import re
import numpy as np
from tqdm import tqdm
from src.noise_models import NOISE_TYPES
from src.fingerprint import build_fingerprint

from concurrent.futures import ProcessPoolExecutor, as_completed


def sample_strength(noise_type, rng, min_strength=0.03, max_strength=0.15):
    """
    Sample a noise strength for a given noise type.

    All noise types currently share the same uniform sampling range.
    This is kept as a separate function so noise-type-specific
    strength ranges can be added later without changing call sites.

    Args:
        noise_type: Noise channel name (unused for now).
        rng: NumPy random Generator.
        min_strength: Lower bound of the sampling range.
        max_strength: Upper bound of the sampling range.

    Returns:
        A single sampled strength value.
    """
    return rng.uniform(min_strength, max_strength)


def generate_one_sample(task):
    """
    Generate one fingerprint sample from a packed task tuple.

    Defined as a standalone function (rather than a closure) so it
    can be pickled and run inside a ProcessPoolExecutor worker.

    Args:
        task: Tuple of (sample_id, noise_type, class_id, strength,
            shots, sample_seed, n_qubits, num_qaoa_probes,
            include_simple_probes, include_derived_features).

    Returns:
        Tuple of (sample_id, fingerprint, class_id, meta).

    Raises:
        Whatever build_fingerprint raises -- callers that want to
        skip failures instead of crashing should use
        safe_generate_one_sample instead.
    """

    (
        sample_id,
        noise_type,
        class_id,
        strength,
        shots,
        sample_seed,
        n_qubits,
        num_qaoa_probes,
        include_simple_probes,
        include_derived_features,
    ) = task

    fingerprint = build_fingerprint(
        noise_type=noise_type,
        strength=strength,
        n_qubits=n_qubits,
        shots=shots,
        seed=sample_seed,
        num_qaoa_probes=num_qaoa_probes,
        include_simple_probes=include_simple_probes,
        include_derived_features=include_derived_features,
    )

    meta = {
        "sample_id": sample_id,
        "noise_type": noise_type,
        "label": class_id,
        "strength": float(strength),
        "shots": shots,
        "seed": sample_seed,
        "n_qubits": n_qubits,
        "num_qaoa_probes": num_qaoa_probes,
        "include_simple_probes": include_simple_probes,
        "include_derived_features": include_derived_features,
    }

    return sample_id, fingerprint, class_id, meta


def safe_generate_one_sample(task):
    """
    Wraps generate_one_sample so a single failing task (e.g. the rare
    "Kraus is empty" / Hermitian eigensolver crash) doesn't kill the
    whole dataset generation run.

    Returns the normal (sample_id, fingerprint, class_id, meta) tuple
    on success. On failure, returns (sample_id, None, class_id, meta)
    where meta contains the failure details instead -- callers must
    check `fingerprint is None` to detect a skipped sample.
    """
    (
        sample_id,
        noise_type,
        class_id,
        strength,
        shots,
        sample_seed,
        n_qubits,
        num_qaoa_probes,
        include_simple_probes,
        include_derived_features,
    ) = task

    try:
        return generate_one_sample(task)
    except Exception as e:
        meta = {
            "sample_id": sample_id,
            "noise_type": noise_type,
            "label": class_id,
            "strength": float(strength),
            "shots": shots,
            "seed": sample_seed,
            "n_qubits": n_qubits,
            "num_qaoa_probes": num_qaoa_probes,
            "include_simple_probes": include_simple_probes,
            "include_derived_features": include_derived_features,
            "error": str(e),
        }
        return sample_id, None, class_id, meta


def ensure_output_directory(output_path):
    """Create the parent directory for an output file if needed."""
    Path(output_path).expanduser().parent.mkdir(parents=True, exist_ok=True)


def checkpoint_path_for(output_path, completed_count):
    """Return the checkpoint path for a completed-sample count."""
    output_path = Path(output_path).expanduser()
    suffix = output_path.suffix or ".npz"
    return output_path.with_name(
        f"{output_path.stem}_checkpoint_{completed_count}{suffix}"
    )


def find_latest_checkpoint(output_path):
    """Find the checkpoint with the largest saved-result count."""
    output_path = Path(output_path).expanduser()
    suffix = output_path.suffix or ".npz"
    pattern = f"{output_path.stem}_checkpoint_*{suffix}"
    regex = re.compile(
        rf"^{re.escape(output_path.stem)}_checkpoint_(\d+){re.escape(suffix)}$"
    )

    candidates = []
    for path in output_path.parent.glob(pattern):
        match = regex.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))

    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def load_checkpoint(checkpoint_path, expected_noise_types, expected_config):
    """Load a resumable checkpoint and validate its label configuration."""
    with np.load(checkpoint_path, allow_pickle=True) as data:
        required = {"X", "y", "labels", "sample_ids", "meta", "config"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"Checkpoint {checkpoint_path} is not resumable; missing keys: "
                f"{sorted(missing)}. Older checkpoints must be regenerated."
            )

        labels = data["labels"].tolist()
        if labels != list(expected_noise_types):
            raise ValueError(
                "Checkpoint noise types do not match this run. "
                f"Checkpoint: {labels}; requested: {list(expected_noise_types)}"
            )

        saved_config = data["config"].item()
        if saved_config != expected_config:
            differences = {
                key: (saved_config.get(key), expected_config.get(key))
                for key in sorted(set(saved_config) | set(expected_config))
                if saved_config.get(key) != expected_config.get(key)
            }
            raise ValueError(
                "Checkpoint settings do not match this run. "
                f"Differences (checkpoint, requested): {differences}"
            )

        X = data["X"]
        y = data["y"]
        sample_ids = data["sample_ids"]
        meta = data["meta"]

    if not (len(X) == len(y) == len(sample_ids) == len(meta)):
        raise ValueError(f"Checkpoint {checkpoint_path} contains inconsistent array lengths.")

    results = [
        (int(sample_id), fingerprint, int(class_id), metadata)
        for sample_id, fingerprint, class_id, metadata in zip(
            sample_ids, X, y, meta
        )
    ]
    return results


def save_checkpoint(results, output_path, noise_types, total, config):
    """
    Save a partial dataset checkpoint to disk.

    Args:
        results: List of (sample_id, fingerprint, class_id, meta) tuples
            collected so far. Only successful (non-None fingerprint)
            results should be passed in.
        output_path: Final output path, used to derive checkpoint filename.
        noise_types: List of noise type names.
        total: Total number of samples expected (for progress reporting).
        config: Generation settings required to validate a future resume.
    """
    n = len(results)
    partial = sorted(results, key=lambda x: x[0])
    X_partial = np.array([r[1] for r in partial], dtype=np.float32)
    y_partial = np.array([r[2] for r in partial], dtype=np.int64)
    sample_ids_partial = np.array([r[0] for r in partial], dtype=np.int64)
    meta_partial = np.array([r[3] for r in partial], dtype=object)

    ensure_output_directory(output_path)
    checkpoint_path = checkpoint_path_for(output_path, n)
    np.savez_compressed(
        checkpoint_path,
        X=X_partial,
        y=y_partial,
        labels=np.array(noise_types, dtype=object),
        sample_ids=sample_ids_partial,
        meta=meta_partial,
        config=np.array(config, dtype=object),
    )
    print(f"\nCheckpoint saved: {checkpoint_path} ({n}/{total} samples)")


def generate_dataset(
    output_path,
    samples_per_class=1500,
    shots=200,
    seed=42,
    n_qubits=3,
    num_qaoa_probes=5,
    min_strength=0.05,
    max_strength=0.15,
    include_simple_probes=True,
    include_derived_features=True,
    noise_types=None,
    num_workers=4,
    checkpoint_every=1000,
    resume=False,
):
    """
    Generate the full labeled noise fingerprint dataset and save it
    to output_path as a compressed .npz file.

    Args:
        output_path: Path to save the .npz file.
        samples_per_class: Number of examples per noise class.
        shots: Number of classical shadow measurements per probe.
        seed: Global random seed.
        n_qubits: Number of qubits.
        num_qaoa_probes: Number of QAOA-style probe circuits per fingerprint.
        min_strength: Minimum noise strength.
        max_strength: Maximum noise strength.
        include_simple_probes: Whether to include simple structured probes.
        include_derived_features: Whether to append derived features.
        noise_types: List of noise types to include.
        num_workers: Number of parallel worker processes.
        checkpoint_every: Save a checkpoint every this many completed samples.
        resume: Resume from the latest compatible checkpoint if available.
    """

    output_path = Path(output_path).expanduser()
    ensure_output_directory(output_path)

    rng = np.random.default_rng(seed)

    if noise_types is None:
        noise_types = NOISE_TYPES

    label_to_id = {
        noise_type: idx
        for idx, noise_type in enumerate(noise_types)
    }

    total = samples_per_class * len(noise_types)

    resume_config = {
        "samples_per_class": int(samples_per_class),
        "shots": int(shots),
        "seed": int(seed),
        "n_qubits": int(n_qubits),
        "num_qaoa_probes": int(num_qaoa_probes),
        "min_strength": float(min_strength),
        "max_strength": float(max_strength),
        "include_simple_probes": bool(include_simple_probes),
        "include_derived_features": bool(include_derived_features),
        "noise_types": list(noise_types),
    }

    print("=" * 80)
    print("Generating quantum noise fingerprint dataset")
    print("=" * 80)
    print(f"Noise types          : {noise_types}")
    print(f"Samples per class    : {samples_per_class}")
    print(f"Total samples        : {total}")
    print(f"Shots per probe      : {shots}")
    print(f"Number of qubits     : {n_qubits}")
    print(f"QAOA probes/sample   : {num_qaoa_probes}")
    print(f"Simple probes        : {include_simple_probes}")
    print(f"Derived features     : {include_derived_features}")
    print(f"Strength range       : [{min_strength}, {max_strength}]")
    print(f"Output path          : {output_path}")
    print(f"Checkpoint every     : {checkpoint_every} samples")
    print(f"Resume enabled       : {resume}")
    print("=" * 80)

    tasks = []
    sample_id = 0

    for noise_type in noise_types:
        class_id = label_to_id[noise_type]

        for _ in range(samples_per_class):
            strength = sample_strength(
                noise_type=noise_type,
                rng=rng,
                min_strength=min_strength,
                max_strength=max_strength,
            )

            sample_seed = seed + sample_id * 17

            task = (
                sample_id,
                noise_type,
                class_id,
                strength,
                shots,
                sample_seed,
                n_qubits,
                num_qaoa_probes,
                include_simple_probes,
                include_derived_features,
            )

            tasks.append(task)
            sample_id += 1

    print(f"Using num_workers    : {num_workers}")

    results = []
    failed_tasks = []

    if resume:
        checkpoint_path = find_latest_checkpoint(output_path)
        if checkpoint_path is None:
            print("Resume requested, but no checkpoint was found. Starting from scratch.")
        else:
            results = load_checkpoint(checkpoint_path, noise_types, resume_config)
            completed_ids = {r[0] for r in results}
            tasks = [task for task in tasks if task[0] not in completed_ids]
            print(f"Resuming from       : {checkpoint_path}")
            print(f"Loaded samples      : {len(results)}")
            print(f"Remaining tasks     : {len(tasks)}")

    if not tasks:
        print("All requested samples are already present in the checkpoint.")

    last_checkpoint_count = len(results)

    if num_workers == 1:
        for task in tqdm(tasks, desc="Generating samples"):
            r = safe_generate_one_sample(task)
            if r[1] is None:
                failed_tasks.append(r[3])
                print(f"\n  SKIPPED (failed): sample_id={r[3]['sample_id']} "
                      f"noise_type={r[3]['noise_type']} strength={r[3]['strength']:.6f} "
                      f"seed={r[3]['seed']} -> {r[3]['error']}")
            else:
                results.append(r)

            if (
                checkpoint_every > 0
                and len(results) > last_checkpoint_count
                and len(results) % checkpoint_every == 0
            ):
                save_checkpoint(results, output_path, noise_types, total, resume_config)
                last_checkpoint_count = len(results)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(safe_generate_one_sample, task)
                for task in tasks
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Generating samples",
            ):
                r = future.result()  # safe_generate_one_sample never raises
                if r[1] is None:
                    failed_tasks.append(r[3])
                    print(f"\n  SKIPPED (failed): sample_id={r[3]['sample_id']} "
                          f"noise_type={r[3]['noise_type']} strength={r[3]['strength']:.6f} "
                          f"seed={r[3]['seed']} -> {r[3]['error']}")
                else:
                    results.append(r)

                if (
                    checkpoint_every > 0
                    and len(results) > last_checkpoint_count
                    and len(results) % checkpoint_every == 0
                ):
                    save_checkpoint(results, output_path, noise_types, total, resume_config)
                    last_checkpoint_count = len(results)

    # Sort results by sample_id so dataset order is deterministic
    # regardless of worker completion order.
    results = sorted(results, key=lambda x: x[0])

    X = np.array([r[1] for r in results], dtype=np.float32)
    y = np.array([r[2] for r in results], dtype=np.int64)
    labels = np.array(noise_types, dtype=object)
    meta = np.array([r[3] for r in results], dtype=object)

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        labels=labels,
        meta=meta,
        seed=seed,
        samples_per_class=samples_per_class,
        shots=shots,
        n_qubits=n_qubits,
        num_qaoa_probes=num_qaoa_probes,
        min_strength=min_strength,
        max_strength=max_strength,
        include_simple_probes=include_simple_probes,
        include_derived_features=include_derived_features,
    )

    if failed_tasks:
        failed_log_path = output_path.with_name(f"{output_path.stem}_failed_tasks.json")
        with open(failed_log_path, "w") as f:
            json.dump(failed_tasks, f, indent=2)

    print("\n" + "=" * 80)
    print("Dataset generation complete")
    print("=" * 80)
    print(f"Saved to       : {output_path}")
    print(f"X shape        : {X.shape}")
    print(f"y shape        : {y.shape}")
    print(f"Labels         : {labels.tolist()}")
    print(f"Feature dim    : {X.shape[1]}")

    print()
    print("Per-class sample counts:")
    for class_id, noise_type in enumerate(noise_types):
        count = int(np.sum(y == class_id))
        flag = "  <-- SHORT" if count < samples_per_class else ""
        print(f"  class {class_id} ({noise_type:<25}): {count} samples{flag}")

    if failed_tasks:
        print()
        print("=" * 80)
        print(f"WARNING: {len(failed_tasks)} task(s) failed and were skipped.")
        print(f"Details saved to: {failed_log_path}")
        print("Use recover_failed_tasks.py to retry/recover them, then")
        print("merge_recovered.py to fold them into this dataset.")
        print("=" * 80)

    print("=" * 80)


def parse_noise_types(noise_types_arg):
    """
    Parse a comma-separated noise type string into a validated list.

    Args:
        noise_types_arg: Either "all", or a comma-separated string
            such as "depolarizing,amplitude_damping,phase_damping".

    Returns:
        List of validated noise type names.

    Raises:
        ValueError: If any provided noise type is not recognized.
    """

    if noise_types_arg is None or noise_types_arg.lower() == "all":
        return NOISE_TYPES

    noise_types = [
        item.strip()
        for item in noise_types_arg.split(",")
        if item.strip()
    ]

    invalid = [
        noise_type
        for noise_type in noise_types
        if noise_type not in NOISE_TYPES
    ]

    if invalid:
        raise ValueError(
            f"Invalid noise types: {invalid}\n"
            f"Available noise types: {NOISE_TYPES}"
        )

    return noise_types


def main():
    parser = argparse.ArgumentParser(
        description="Generate quantum noise fingerprint dataset."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/noise_fingerprints_shadow_structured.npz",
        help="Output .npz path.",
    )

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=1500,
        help="Number of samples per noise type.",
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=200,
        help="Number of classical shadow shots per probe.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--n-qubits",
        type=int,
        default=3,
        help="Number of qubits.",
    )

    parser.add_argument(
        "--num-qaoa-probes",
        type=int,
        default=5,
        help="Number of QAOA probe circuits per sample.",
    )

    parser.add_argument(
        "--min-strength",
        type=float,
        default=0.05,
        help="Minimum noise strength.",
    )

    parser.add_argument(
        "--max-strength",
        type=float,
        default=0.25,
        help="Maximum noise strength.",
    )

    parser.add_argument(
        "--noise-types",
        type=str,
        default="all",
        help=(
            "Comma-separated noise types, or 'all'. "
            "Example: depolarizing,amplitude_damping,phase_damping"
        ),
    )

    parser.add_argument(
        "--no-simple-probes",
        action="store_true",
        help="Disable simple probes and use only QAOA probes.",
    )

    parser.add_argument(
        "--no-derived-features",
        action="store_true",
        help="Disable derived features and use only raw observables.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for dataset generation.",
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        help="Save a checkpoint every this many completed samples.",
    )


    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest compatible checkpoint for this output path.",
    )

    args = parser.parse_args()

    noise_types = parse_noise_types(args.noise_types)

    generate_dataset(
        output_path=args.output,
        samples_per_class=args.samples_per_class,
        shots=args.shots,
        seed=args.seed,
        n_qubits=args.n_qubits,
        num_qaoa_probes=args.num_qaoa_probes,
        min_strength=args.min_strength,
        max_strength=args.max_strength,
        include_simple_probes=not args.no_simple_probes,
        include_derived_features=not args.no_derived_features,
        noise_types=noise_types,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()