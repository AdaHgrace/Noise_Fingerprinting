"""
observables.py

Pauli observables for quantum noise fingerprinting, generalized to
any number of qubits.

Pattern:
    - Single-qubit observables: X/Y/Z on each individual qubit
      position, identity elsewhere. Count: 3 * n_qubits.
    - Two-qubit same-axis correlations: X/Y/Z applied identically to
      every PAIR of qubits (all C(n_qubits, 2) pairs, not just
      adjacent ones), identity elsewhere. Count: 3 * C(n_qubits, 2).

    Total observables for n qubits: 3*n + 3*C(n,2) = 3*n*(n+1)/2.
    For n=3 this reproduces your original 18 observables exactly.

Growth is QUADRATIC in n_qubits.

Use generate_observables(n_qubits) to get the right observable set
and groups for a given qubit count. The module-level OBSERVABLES /
OBSERVABLE_GROUPS below remain as the n_qubits=3 case, for backward
compatibility with any code that imports them directly.
"""

from itertools import combinations

def generate_observables(n_qubits: int):
    """
    Generate the full observable set and groups for a given number
    of qubits, following the same pattern as the original 3-qubit set:
    all single-qubit X/Y/Z observables, plus all pairwise same-axis
    two-qubit correlations.

    Args:
        n_qubits: Number of qubits.

    Returns:
        (observables, observable_groups) where:
            observables: list of Pauli strings, length n_qubits each.
            observable_groups: dict with keys "x_like", "y_like",
                "z_like", "mixed", "single_qubit", "two_qubit".
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1.")

    axes = ["X", "Y", "Z"]

    def make_string(positions_and_axis, n):
        """positions_and_axis: dict {qubit_index: axis_char}"""
        chars = ["I"] * n
        for pos, axis in positions_and_axis.items():
            chars[pos] = axis
        return "".join(chars)

    single_qubit = []
    for axis in axes:
        for pos in range(n_qubits):
            single_qubit.append(make_string({pos: axis}, n_qubits))

    two_qubit = []
    if n_qubits >= 2:
        for axis in axes:
            for i, j in combinations(range(n_qubits), 2):
                two_qubit.append(make_string({i: axis, j: axis}, n_qubits))

    observables = single_qubit + two_qubit

    groups = {
        "x_like": [obs for obs in observables if obs.count("X") > 0],
        "y_like": [obs for obs in observables if obs.count("Y") > 0],
        "z_like": [obs for obs in observables if obs.count("Z") > 0],
        "mixed": [],  # reserved for future mixed-axis observables (e.g. "XYI")
        "single_qubit": single_qubit,
        "two_qubit": two_qubit,
    }

    return observables, groups


def validate_observables(observables):
    allowed = {"I", "X", "Y", "Z"}

    if len(observables) == 0:
        raise ValueError("Observable list is empty.")

    n_qubits = len(observables[0])

    for obs in observables:
        if len(obs) != n_qubits:
            raise ValueError(f"Observable {obs} has inconsistent length.")

        for char in obs:
            if char not in allowed:
                raise ValueError(f"Invalid Pauli character {char} in {obs}.")

    return True


def get_num_qubits_from_observable(obs: str) -> int:
    return len(obs)

# Backward-compatible module-level constants for the original 3-qubit
# case, so any existing code that does
# `from src.observables import OBSERVABLES, OBSERVABLE_GROUPS`
# continues to work unchanged.
OBSERVABLES, OBSERVABLE_GROUPS = generate_observables(3)