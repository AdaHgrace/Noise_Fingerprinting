"""
shadow.py

Classical shadow tomography utilities for the Shadow-Based Noise
Fingerprinting pipeline described in:

    "Shadow-Based Noise Fingerprinting for Quantum Processors"
    Vridhi Jain, Lei Zhang (2026)

This module implements a randomized Pauli classical shadow measurement
pipeline for estimating Pauli observable expectation values from a
quantum circuit executed on a (potentially noisy) device.

Pipeline overview
-----------------
For each of ``shots`` independent measurements:

1. Randomly sample a Pauli basis for every qubit from {X, Y, Z}.
2. Rotate the circuit into that basis and measure all qubits,
   obtaining a classical bitstring.
3. Store the chosen basis string (e.g. "XZY") and the measured
   bitstring (e.g. "010").

Observable estimation
---------------------
For a Pauli observable P (e.g. "XIY"), a single shot contributes to
the estimate only if every non-identity Pauli in P was measured in the
matching basis.  The standard classical shadow inverse channel factor
3^k is applied, where k is the number of non-identity Paulis in P.

Example::

    Observable: "XIZ"
    Basis:      "XZZ"

    qubit 0: X matches X  → contributes
    qubit 1: I is ignored → contributes (identity always matches)
    qubit 2: Z matches Z  → contributes

    This shot CAN contribute to estimating "XIZ".

This approach lets many observables be estimated from the same
randomized measurement data, avoiding a separate circuit execution per
observable.

Reference
---------
Huang, H.-Y., Kueng, R., & Preskill, J. (2020).
Predicting many properties of a quantum system from very few
measurements. *Nature Physics*, 16(10), 1050–1057.
"""

import numpy as np

from qiskit import transpile
from qiskit_aer import AerSimulator


PAULI_BASES = ["X", "Y", "Z"]


def _validate_pauli_string(pauli_string: str):
    """Raise if pauli_string is not a non-empty string of I/X/Y/Z characters."""

    allowed = {"I", "X", "Y", "Z"}

    if not isinstance(pauli_string, str):
        raise TypeError("Pauli observable must be a string.")

    if len(pauli_string) == 0:
        raise ValueError("Pauli observable cannot be empty.")

    for char in pauli_string:
        if char not in allowed:
            raise ValueError(
                f"Invalid Pauli character '{char}' in observable '{pauli_string}'."
            )


def _rotate_into_basis(circuit, basis_string: str):
    """
    Return a copy of circuit with single-qubit basis rotations and measure_all appended.

    Z basis: no rotation. X basis: H gate. Y basis: S† then H.
    """

    n_qubits = circuit.num_qubits

    if len(basis_string) != n_qubits:
        raise ValueError(
            f"Basis string length {len(basis_string)} does not match "
            f"number of qubits {n_qubits}."
        )

    qc = circuit.copy()

    for q, basis in enumerate(basis_string):
        if basis == "X":
            qc.h(q)
        elif basis == "Y":
            qc.sdg(q)
            qc.h(q)
        elif basis == "Z":
            pass
        else:
            raise ValueError(f"Invalid basis '{basis}'. Must be X, Y, or Z.")

    qc.measure_all()

    return qc


def generate_random_bases(
    n_qubits: int,
    shots: int,
    seed: int = 42,
):
    """
    Sample random Pauli measurement bases for all shots.

    Args:
        n_qubits: Number of qubits in the circuit.
        shots: Number of shadow measurements (one basis per shot).
        seed: Random seed for reproducibility.

    Returns:
        NumPy array of shape ``(shots, n_qubits)`` with entries in
        ``{"X", "Y", "Z"}``.
    """

    rng = np.random.default_rng(seed)

    bases = rng.choice(
        PAULI_BASES,
        size=(shots, n_qubits),
        replace=True,
    )

    return bases


def basis_array_to_strings(bases):
    """
    Convert a ``(shots, n_qubits)`` basis array to a list of strings.

    Example::

        [["X", "Z"], ["Y", "X"]] → ["XZ", "YX"]

    Args:
        bases: NumPy array of shape ``(shots, n_qubits)``.

    Returns:
        List of ``shots`` basis strings.
    """

    return ["".join(row.tolist()) for row in bases]


def _sample_one_basis(
    circuit,
    basis_string: str,
    noise_model=None,
    seed: int = 42,
):
    """
    Execute circuit in basis_string for a single shot and return the measured bitstring.

    Uses shots=1 so each measurement can use an independently sampled random basis,
    matching the classical shadow protocol. bitstring[q] corresponds to qubit q.
    """

    measured_circuit = _rotate_into_basis(
        circuit=circuit,
        basis_string=basis_string,
    )

    simulator = AerSimulator(
        noise_model=noise_model,
        seed_simulator=seed,
    )

    compiled = transpile(measured_circuit, simulator)

    result = simulator.run(
        compiled,
        shots=1,
    ).result()

    counts = result.get_counts()

    # Qiskit returns bitstrings with qubit order reversed (classical
    # register order). Reverse so bitstring[q] corresponds to qubit q.
    raw_bitstring = next(iter(counts.keys()))
    bitstring = raw_bitstring.replace(" ", "")[::-1]

    return bitstring


def run_shadow_tomography(
    circuit,
    noise_model=None,
    shots: int = 200,
    seed: int = 42,
):
    """
    Run randomized Pauli classical shadow tomography on ``circuit``.

    Samples ``shots`` independent random Pauli bases, executes the
    circuit in each basis with a unique per-shot seed, and collects
    all measurement outcomes.

    Args:
        circuit: Qiskit ``QuantumCircuit`` without measurements.
        noise_model: Optional Qiskit Aer ``NoiseModel``.
        shots: Number of randomized shadow measurements.
        seed: Global random seed; per-shot seeds are derived as
            ``seed + i`` for shot index ``i``.

    Returns:
        Dictionary with keys:

        * ``"bases"`` — array of shape ``(shots, n_qubits)``.
        * ``"basis_strings"`` — list of basis strings.
        * ``"bitstrings"`` — list of measured bitstrings in qubit order.
        * ``"n_qubits"`` — number of qubits.
        * ``"shots"`` — number of shadow measurements.
    """

    n_qubits = circuit.num_qubits

    bases = generate_random_bases(
        n_qubits=n_qubits,
        shots=shots,
        seed=seed,
    )

    basis_strings = basis_array_to_strings(bases)

    bitstrings = []

    for i, basis_string in enumerate(basis_strings):
        bitstring = _sample_one_basis(
            circuit=circuit,
            basis_string=basis_string,
            noise_model=noise_model,
            seed=seed + i,
        )

        bitstrings.append(bitstring)

    return {
        "bases": bases,
        "basis_strings": basis_strings,
        "bitstrings": bitstrings,
        "n_qubits": n_qubits,
        "shots": shots,
    }


def _shot_matches_observable(
    basis_string: str,
    observable: str,
):
    """Return True if every non-identity Pauli in observable was measured in the matching basis."""

    for basis, pauli in zip(basis_string, observable):
        if pauli == "I":
            continue
        if basis != pauli:
            return False

    return True


def _pauli_eigenvalue_from_bitstring(
    bitstring: str,
    observable: str,
):
    """Return the product of per-qubit eigenvalues (+1 for bit 0, -1 for bit 1) at non-identity positions."""

    eigenvalue = 1

    for bit, pauli in zip(bitstring, observable):
        if pauli == "I":
            continue
        if bit == "1":
            eigenvalue *= -1
        elif bit != "0":
            raise ValueError(f"Invalid measured bit '{bit}'.")

    return eigenvalue


def estimate_pauli_from_shadow(
    shadow_data,
    observable: str,
):
    """
    Estimate the expectation value of a Pauli observable from shadow data.

    Only shots whose basis is compatible with ``observable`` contribute.
    Each contributing shot is weighted by the classical shadow inverse
    channel factor ``3^k``, where ``k`` is the number of non-identity
    Paulis in the observable.  The result is clipped to ``[-1, 1]`` to
    correct for finite-shot numerical noise.

    Args:
        shadow_data: Output of :func:`run_shadow_tomography`.
        observable: Pauli string, e.g. ``"XI"``, ``"ZZ"``, ``"XYZ"``.

    Returns:
        Estimated expectation value in ``[-1, 1]``.

    Raises:
        ValueError: If ``observable`` length does not match ``n_qubits``.
    """

    _validate_pauli_string(observable)

    n_qubits = shadow_data["n_qubits"]

    if len(observable) != n_qubits:
        raise ValueError(
            f"Observable '{observable}' has length {len(observable)}, "
            f"but shadow data has {n_qubits} qubits."
        )

    basis_strings = shadow_data["basis_strings"]
    bitstrings = shadow_data["bitstrings"]
    shots = shadow_data["shots"]

    k = sum(1 for p in observable if p != "I")
    inverse_channel_factor = 3 ** k

    total = 0.0

    for basis_string, bitstring in zip(basis_strings, bitstrings):
        if not _shot_matches_observable(basis_string, observable):
            continue

        total += inverse_channel_factor * _pauli_eigenvalue_from_bitstring(
            bitstring, observable
        )

    expectation = float(np.clip(total / shots, -1.0, 1.0))

    return expectation


def estimate_many_paulis_from_shadow(
    shadow_data,
    observables,
):
    """
    Estimate multiple Pauli observables from the same shadow data.

    Reusing one set of shadow measurements for many observables is
    the core efficiency advantage of classical shadow tomography.

    Args:
        shadow_data: Output of :func:`run_shadow_tomography`.
        observables: Iterable of Pauli strings.

    Returns:
        NumPy array of shape ``(len(observables),)`` with estimated
        expectation values, dtype ``float32``.
    """

    return np.array(
        [estimate_pauli_from_shadow(shadow_data, obs) for obs in observables],
        dtype=np.float32,
    )


def run_shadow_and_estimate(
    circuit,
    observables,
    noise_model=None,
    shots: int = 200,
    seed: int = 42,
):
    """
    Convenience function: run shadow tomography and estimate observables.

    Combines :func:`run_shadow_tomography` and
    :func:`estimate_many_paulis_from_shadow` into a single call.

    Args:
        circuit: Qiskit ``QuantumCircuit`` without measurements.
        observables: Iterable of Pauli strings to estimate.
        noise_model: Optional Qiskit Aer ``NoiseModel``.
        shots: Number of shadow measurements.
        seed: Random seed.

    Returns:
        Tuple of:

        * ``values`` — NumPy array of observable estimates, shape
          ``(len(observables),)``.
        * ``shadow_data`` — raw shadow data dictionary (see
          :func:`run_shadow_tomography`).
    """

    shadow_data = run_shadow_tomography(
        circuit=circuit,
        noise_model=noise_model,
        shots=shots,
        seed=seed,
    )

    values = estimate_many_paulis_from_shadow(
        shadow_data=shadow_data,
        observables=observables,
    )

    return values, shadow_data