"""
observables.py

3-qubit Pauli observables used for classical shadow estimation in the
noise fingerprinting pipeline.

OBSERVABLES contains 18 Pauli strings: single-qubit terms (e.g. "XII")
and two-qubit same-axis correlations (e.g. "XXI"). OBSERVABLE_GROUPS
organizes these into axis-based and locality-based subsets used for
derived feature construction in fingerprint.py.
 
Each sample uses 9 probe circuits (4 simple + 5 QAOA). Per probe:
18 raw observables + 13 derived features = 31 features.
Total feature vector size: 9 x 31 = 279, matching the dimensionality
reported in the accompanying paper.
"""

OBSERVABLES = [
    # Single-qubit observables
    "XII", "YII", "ZII",
    "IXI", "IYI", "IZI",
    "IIX", "IIY", "IIZ",

    # Two-qubit same-axis correlations
    "XXI", "YYI", "ZZI",
    "XIX", "YIY", "ZIZ",
    "IXX", "IYY", "IZZ",
]


OBSERVABLE_GROUPS = {
    "x_like": [
        "XII", "IXI", "IIX",
        "XXI", "XIX", "IXX",
    ],

    "y_like": [
        "YII", "IYI", "IIY",
        "YYI", "YIY", "IYY",
    ],

    "z_like": [
        "ZII", "IZI", "IIZ",
        "ZZI", "ZIZ", "IZZ",
    ],

    # Empty: OBSERVABLES contains no mixed-axis Pauli strings.
    "mixed": [],

    "single_qubit": [
        "XII", "YII", "ZII",
        "IXI", "IYI", "IZI",
        "IIX", "IIY", "IIZ",
    ],

    "two_qubit": [
        "XXI", "YYI", "ZZI",
        "XIX", "YIY", "ZIZ",
        "IXX", "IYY", "IZZ",
    ],
}


def validate_observables(observables):
    """
    Validate a list of Pauli observable strings.

    Checks that the list is non-empty, that every observable has the
    same length (i.e. acts on the same number of qubits), and that
    every character is a valid Pauli label (I, X, Y, or Z).

    Args:
        observables: List of Pauli strings, e.g. ["XII", "ZZI"].

    Returns:
        True if all observables are valid.

    Raises:
        ValueError: If the list is empty, observables have
            inconsistent lengths, or an invalid character is found.
    """
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
    """
    Return the number of qubits implied by a Pauli observable string.

    Args:
        obs: Pauli string, e.g. "XII".

    Returns:
        Length of the string, equal to the number of qubits it acts on.
    """
    return len(obs)