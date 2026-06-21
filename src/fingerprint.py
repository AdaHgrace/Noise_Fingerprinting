"""
fingerprint.py

Builds feature vectors for quantum noise fingerprinting.

Each fingerprint vector concatenates, for every probe circuit:
1. Raw expectation values from classical shadow estimation
2. Derived physics-inspired features: X/Y/Z grouped means, a
   coherence-vs-population ratio, and pairwise group differences
"""

import numpy as np

from src.noise_models import get_noise_model
from src.observables import OBSERVABLES, OBSERVABLE_GROUPS
from src.circuits import get_probe_circuits
from src.shadow import run_shadow_and_estimate


def compute_group_mean(feature_dict, group):
    """
    Mean absolute expectation value over a group of observables.

    Args:
        feature_dict: Dict mapping observable strings to their
            estimated expectation values.
        group: List of observable strings to average over.

    Returns:
        Mean absolute value over the group, or 0.0 if no observable
        in the group is present in feature_dict.
    """
    values = [abs(feature_dict[obs]) for obs in group if obs in feature_dict]

    if len(values) == 0:
        return 0.0

    return float(np.mean(values))


def build_derived_features(raw_values, observables=OBSERVABLES):
    """
    Build physics-inspired derived features from raw expectation values.

    Args:
        raw_values: List or numpy array of expectation values for one
            probe circuit.
        observables: Observable names corresponding to raw_values.

    Returns:
        List of 13 derived features for the given probe.
    """

    eps = 1e-8

    feature_dict = {
        obs: float(val)
        for obs, val in zip(observables, raw_values)
    }

    mean_x = compute_group_mean(feature_dict, OBSERVABLE_GROUPS["x_like"])
    mean_y = compute_group_mean(feature_dict, OBSERVABLE_GROUPS["y_like"])
    mean_z = compute_group_mean(feature_dict, OBSERVABLE_GROUPS["z_like"])
    mean_mixed = compute_group_mean(feature_dict, OBSERVABLE_GROUPS["mixed"])

    coherence_strength = mean_x + mean_y
    population_strength = mean_z

    derived = [
        mean_x,
        mean_y,
        mean_z,
        mean_mixed,

        coherence_strength,
        population_strength,

        mean_x - mean_y,
        mean_z - mean_x,
        mean_z - mean_y,

        coherence_strength / (population_strength + eps),
        population_strength / (coherence_strength + eps),

        mean_mixed / (coherence_strength + eps),
        mean_mixed / (population_strength + eps),
    ]

    return derived


def build_fingerprint(
    noise_type: str,
    strength: float,
    n_qubits: int = 2,
    shots: int = 200,
    seed: int = 42,
    num_qaoa_probes: int = 5,
    include_simple_probes: bool = True,
    include_derived_features: bool = True,
):
    """
    Build one fingerprint vector for a given noise model.

    This is the main entry point used by generate_dataset.py to build
    the labeled training dataset.

    Args:
        noise_type: Noise channel name (see noise_models.NOISE_TYPES).
        strength: Noise strength, in [0, 1].
        n_qubits: Number of qubits.
        shots: Number of classical shadow measurement shots per probe.
        seed: Random seed.
        num_qaoa_probes: Number of QAOA-style probe circuits.
        include_simple_probes: Whether to include the simple structured
            probes (basis states, superposition, Bell state).
        include_derived_features: Whether to append derived features
            after the raw observables for each probe.

    Returns:
        1D numpy array feature vector.
    """

    noise_model = get_noise_model(noise_type, strength)

    probes = get_probe_circuits(
        n_qubits=n_qubits,
        num_qaoa_probes=num_qaoa_probes,
        seed=seed,
        include_simple_probes=include_simple_probes,
    )

    all_features = []

    for probe_idx, (probe_name, circuit) in enumerate(probes):
        raw_values, _ = run_shadow_and_estimate(
            circuit=circuit,
            observables=OBSERVABLES,
            noise_model=noise_model,
            shots=shots,
            seed=seed + 1000 * probe_idx,
        )

        all_features.extend(raw_values)

        if include_derived_features:
            derived_values = build_derived_features(
                raw_values=raw_values,
                observables=OBSERVABLES,
            )
            all_features.extend(derived_values)

    return np.array(all_features, dtype=np.float32)