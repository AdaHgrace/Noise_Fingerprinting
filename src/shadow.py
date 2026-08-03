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
For each of `shots` independent measurements:

1. Randomly sample a Pauli basis for every qubit from {X, Y, Z}.
2. Rotate the circuit into that basis and measure all qubits,
   obtaining a classical bitstring.
3. Store the chosen basis string (e.g. "XZY") and the measured
   bitstring (e.g. "010").

Observable estimation
---------------------
For a Pauli observable P, a single shot contributes to the estimate
only if every non-identity Pauli in P was measured in the matching
basis. The standard classical shadow inverse channel factor 3^k is
applied, where k is the number of non-identity Paulis in P.

Example:
    Observable: "XIZ"
    Basis:      "XZZ"

    qubit 0: X matches X  -> contributes
    qubit 1: I is ignored -> contributes
    qubit 2: Z matches Z  -> contributes

    This shot CAN contribute to estimating "XIZ".

This approach lets many observables be estimated from the same
randomized measurement data, avoiding a separate circuit execution
per observable.

Reference:
    Huang, H.-Y., Kueng, R., & Preskill, J. (2020).
    Predicting many properties of a quantum system from very few
    measurements. Nature Physics, 16(10), 1050-1057.
"""

import numpy as np

from qiskit import transpile
from qiskit_aer import AerSimulator


PAULI_BASES = ["X", "Y", "Z"]


def _validate_pauli_string(pauli_string: str):
    """
    Validate a Pauli string such as 'XIY' or 'ZZ'.
    """

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
    Return a copy of circuit with basis rotations added before measurement.

    Measurement convention:
        Z basis: no rotation
        X basis: H before measurement
        Y basis: Sdg then H before measurement

    Args:
        circuit:
            Qiskit QuantumCircuit without final measurements.
        basis_string:
            String of length n_qubits, e.g. "XYZ".

    Returns:
        New QuantumCircuit with measurement operations.
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
            raise ValueError(f"Invalid basis '{basis}'. Use X, Y, or Z.")

    qc.measure_all()

    return qc


def generate_random_bases(
    n_qubits: int,
    shots: int,
    seed: int = 42,
):
    """
    Generate random Pauli measurement bases.

    Args:
        n_qubits:
            Number of qubits.
        shots:
            Number of classical shadow measurements.
        seed:
            Random seed.

    Returns:
        NumPy array of shape (shots, n_qubits), entries in {"X", "Y", "Z"}.
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
    Convert basis array of shape (shots, n_qubits) into list of strings.

    Example:
        [["X", "Z"], ["Y", "X"]] -> ["XZ", "YX"]
    """

    return ["".join(row.tolist()) for row in bases]


def run_shadow_tomography(
    circuit,
    noise_model=None,
    shots: int = 200,
    seed: int = 42,
):
    """
    Run randomized Pauli classical shadow tomography for one circuit.

    Shots are grouped by unique measurement basis and executed with one
    simulator call per unique basis (shots=<count for that basis>),
    instead of one simulator call per individual shot. This preserves
    the exact same underlying statistics as simulating shot-by-shot --
    only the execution strategy changes.

    Args:
        circuit:
            Qiskit QuantumCircuit without measurement.
        noise_model:
            Optional Qiskit Aer NoiseModel.
        shots:
            Number of randomized shadow measurements.
        seed:
            Random seed.

    Returns:
        Dictionary containing:
            bases:
                Array of shape (shots, n_qubits), entries X/Y/Z.
            basis_strings:
                List of basis strings, one per collected shot.
            bitstrings:
                List of measured bitstrings aligned to qubit order,
                one per collected shot (same order as basis_strings).
            n_qubits:
                Number of qubits.
            shots:
                Number of shots.
    """

    n_qubits = circuit.num_qubits

    bases = generate_random_bases(
        n_qubits=n_qubits,
        shots=shots,
        seed=seed,
    )

    basis_strings = basis_array_to_strings(bases)

    unique_bases, counts_per_basis = np.unique(basis_strings, return_counts=True)

    simulator = AerSimulator(
        noise_model=noise_model,
        seed_simulator=seed,
    )

    all_basis_strings = []
    all_bitstrings = []

    for basis_idx, (basis_string, n_shots_for_basis) in enumerate(
        zip(unique_bases, counts_per_basis)
    ):
        measured_circuit = _rotate_into_basis(
            circuit=circuit,
            basis_string=basis_string,
        )

        compiled = transpile(measured_circuit, simulator)

        result = simulator.run(
            compiled,
            shots=int(n_shots_for_basis),
            seed_simulator=seed + basis_idx,
        ).result()

        counts = result.get_counts()

        for raw_bitstring, multiplicity in counts.items():
            # Qiskit classical bitstrings are returned with qubit order
            # reversed. Reverse so bitstring[q] corresponds to qubit q.
            bitstring = raw_bitstring.replace(" ", "")[::-1]

            all_basis_strings.extend([basis_string] * int(multiplicity))
            all_bitstrings.extend([bitstring] * int(multiplicity))

    shadow_data = {
        "bases": bases,
        "basis_strings": all_basis_strings,
        "bitstrings": all_bitstrings,
        "n_qubits": n_qubits,
        "shots": shots,
    }

    return shadow_data


def _shot_matches_observable(
    basis_string: str,
    observable: str,
):
    """
    Check whether a shadow shot can estimate a Pauli observable.

    A shot matches if every non-identity Pauli in the observable was
    measured in the same basis.

    Example:
        basis      = "XYZ"
        observable = "XIY"

        qubit 0: X matches X
        qubit 1: I ignored
        qubit 2: observable Y but basis Z -> does not match
    """

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
    """
    Compute the Pauli eigenvalue contribution from a measured bitstring.

    For matching basis measurements:
        bit 0 -> eigenvalue +1
        bit 1 -> eigenvalue -1

    For multi-qubit Pauli observables:
        eigenvalue = product of eigenvalues on non-identity positions.
    """

    eigenvalue = 1

    for bit, pauli in zip(bitstring, observable):
        if pauli == "I":
            continue

        if bit == "1":
            eigenvalue *= -1
        elif bit == "0":
            eigenvalue *= 1
        else:
            raise ValueError(f"Invalid measured bit '{bit}'.")

    return eigenvalue


def estimate_pauli_from_shadow(
    shadow_data,
    observable: str,
):
    """
    Estimate expectation value of a Pauli observable from shadow data.

    Args:
        shadow_data:
            Output of run_shadow_tomography.
        observable:
            Pauli string, e.g. "XI", "ZZ", "XYZ".

    Returns:
        Estimated expectation value <observable>.
    """

    _validate_pauli_string(observable)

    n_qubits = shadow_data["n_qubits"]

    if len(observable) != n_qubits:
        raise ValueError(
            f"Observable {observable} has length {len(observable)}, "
            f"but shadow data has {n_qubits} qubits."
        )

    basis_strings = shadow_data["basis_strings"]
    bitstrings = shadow_data["bitstrings"]
    shots = shadow_data["shots"]

    # Weight is 3^k, where k is the number of non-identity Paulis.
    k = sum(1 for p in observable if p != "I")
    inverse_channel_factor = 3 ** k

    total = 0.0

    for basis_string, bitstring in zip(basis_strings, bitstrings):
        if not _shot_matches_observable(
            basis_string=basis_string,
            observable=observable,
        ):
            continue

        eigenvalue = _pauli_eigenvalue_from_bitstring(
            bitstring=bitstring,
            observable=observable,
        )

        total += inverse_channel_factor * eigenvalue

    expectation = total / shots

    # Numerical clipping because finite shots can sometimes create
    # slightly noisy estimates outside [-1, 1].
    expectation = float(np.clip(expectation, -1.0, 1.0))

    return expectation


def estimate_many_paulis_from_shadow(
    shadow_data,
    observables,
):
    """
    Estimate many Pauli observables from the same shadow data.

    This is the main advantage of classical shadows.
    """

    values = []

    for observable in observables:
        value = estimate_pauli_from_shadow(
            shadow_data=shadow_data,
            observable=observable,
        )
        values.append(value)

    return np.array(values, dtype=np.float32)


def run_shadow_and_estimate(
    circuit,
    observables,
    noise_model=None,
    shots: int = 200,
    seed: int = 42,
):
    """
    Convenience function:
        1. Run classical shadow tomography
        2. Estimate all requested observables

    Returns:
        values:
            NumPy array of observable estimates.
        shadow_data:
            Raw shadow data dictionary.
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


# Backward-compatible alias from your older code style.
def estimate_pauli_from_shots(shadow_data, observable: str):
    """
    Alias for compatibility with older scripts.
    """
    return estimate_pauli_from_shadow(
        shadow_data=shadow_data,
        observable=observable,
    )