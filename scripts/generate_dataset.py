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

Usage (run from the project root):
    python3 -m scripts.generate_dataset \\
        --output data/your_dataset.npz \\
        --samples-per-class 1000 \\
        --shots 200 \\
        --n-qubits 3 \\
        --num-qaoa-probes 5 \\
        --num-workers 4 \\
        --noise-types all
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

from src.noise_models import NOISE_TYPES
from src.fingerprint import build_fingerprint

from concurrent.futures import ProcessPoolExecutor, as_completed


def sample_strength(noise_type, rng, min_strength=0.01, max_strength=0.15):
    """
    Sample a noise strength for a given noise type.

    All noise types currently share the same uniform sampling range.
    This is kept as a separate function so noise-type-specific
    strength ranges can be added later without changing call sites.

    Args:
        noise_type: Noise channel name (unused for now, see above).
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


def generate_dataset(
    output_path,
    samples_per_class=500,
    shots=200,
    seed=42,
    n_qubits=3,
    num_qaoa_probes=5,
    min_strength=0.01,
    max_strength=0.15,
    include_simple_probes=True,
    include_derived_features=True,
    noise_types=None,
    num_workers=4,
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
        num_qaoa_probes: Number of QAOA-style probe circuits per
            fingerprint.
        min_strength: Minimum noise strength.
        max_strength: Maximum noise strength.
        include_simple_probes: Whether to include the simple
            structured probes (basis states, superposition, Bell).
        include_derived_features: Whether to append derived
            group/ratio features.
        noise_types: List of noise types to include. Defaults to
            all types in NOISE_TYPES if not provided.
        num_workers: Number of parallel worker processes. Use 1 to
            run sequentially without multiprocessing.
    """

    rng = np.random.default_rng(seed)

    if noise_types is None:
        noise_types = NOISE_TYPES

    X = []
    y = []
    meta = []

    label_to_id = {
        noise_type: idx
        for idx, noise_type in enumerate(noise_types)
    }

    total = samples_per_class * len(noise_types)

    print("=" * 80)
    print("Generating quantum noise fingerprint dataset")
    print("=" * 80)
    print(f"Noise types          : {noise_types}")
    print(f"Samples per class   : {samples_per_class}")
    print(f"Total samples       : {total}")
    print(f"Shots per probe     : {shots}")
    print(f"Number of qubits    : {n_qubits}")
    print(f"QAOA probes/sample  : {num_qaoa_probes}")
    print(f"Simple probes       : {include_simple_probes}")
    print(f"Derived features    : {include_derived_features}")
    print(f"Strength range      : [{min_strength}, {max_strength}]")
    print(f"Output path         : {output_path}")
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

    print(f"Using num_workers     : {num_workers}")

    results = []

    if num_workers == 1:
        for task in tqdm(tasks, desc="Generating samples"):
            results.append(generate_one_sample(task))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(generate_one_sample, task)
                for task in tasks
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Generating samples",
            ):
                results.append(future.result())

    # Sort by sample_id, since ProcessPoolExecutor completion order
    # is not guaranteed to match submission order. This keeps dataset
    # ordering deterministic regardless of num_workers.
    results = sorted(results, key=lambda x: x[0])

    for sample_id, fingerprint, class_id, sample_meta in results:
        X.append(fingerprint)
        y.append(class_id)
        meta.append(sample_meta)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    labels = np.array(noise_types, dtype=object)
    meta = np.array(meta, dtype=object)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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

    print("\n" + "=" * 80)
    print("Dataset generation complete")
    print("=" * 80)
    print(f"Saved to       : {output_path}")
    print(f"X shape        : {X.shape}")
    print(f"y shape        : {y.shape}")
    print(f"Labels         : {labels.tolist()}")
    print(f"Feature dim    : {X.shape[1]}")
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
        default=500,
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
        default=0.01,
        help="Minimum noise strength.",
    )

    parser.add_argument(
        "--max-strength",
        type=float,
        default=0.15,
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
    )


if __name__ == "__main__":
    main()