"""
noise_models.py

Defines Qiskit Aer noise models for the noise fingerprinting project.
"""

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    phase_amplitude_damping_error,
    thermal_relaxation_error,
    pauli_error,
    reset_error,
    ReadoutError,
)


NOISE_TYPES = [
    "depolarizing",
    "amplitude_damping",
    "phase_damping",
    "phase_amplitude_damping",
    "thermal_relaxation",
    "bit_flip",
    "phase_flip",
    "pauli_asymmetric",
    "readout",
    "reset",
]


def get_noise_model(noise_type: str, strength: float) -> NoiseModel:
    """
    Create a Qiskit Aer NoiseModel.

    Args:
        noise_type:
            Type of noise channel.
        strength:
            Noise strength / probability. Usually between 0 and 1.

    Returns:
        Qiskit Aer NoiseModel.
    """

    if noise_type not in NOISE_TYPES:
        raise ValueError(
            f"Unknown noise type: {noise_type}. "
            f"Available types: {NOISE_TYPES}"
        )

    if not (0 <= strength <= 1):
        raise ValueError("strength must be between 0 and 1.")

    noise_model = NoiseModel()

    one_qubit_gates = ["x", "y", "z", "h", "rx", "ry", "rz", "sx"]
    two_qubit_gates = ["cx"]

    if noise_type == "depolarizing":
        error_1q = depolarizing_error(strength, 1)
        error_2q = depolarizing_error(strength, 2)

        noise_model.add_all_qubit_quantum_error(error_1q, one_qubit_gates)
        noise_model.add_all_qubit_quantum_error(error_2q, two_qubit_gates)

    elif noise_type == "amplitude_damping":
        error_1q = amplitude_damping_error(strength)
        error_2q = error_1q.tensor(error_1q)

        noise_model.add_all_qubit_quantum_error(error_1q, one_qubit_gates)
        noise_model.add_all_qubit_quantum_error(error_2q, two_qubit_gates)

    elif noise_type == "phase_damping":
        error_1q = phase_damping_error(strength)
        error_2q = error_1q.tensor(error_1q)

        noise_model.add_all_qubit_quantum_error(error_1q, one_qubit_gates)
        noise_model.add_all_qubit_quantum_error(error_2q, two_qubit_gates)

    elif noise_type == "phase_amplitude_damping":
        # Split the strength budget between amplitude and phase damping
        # so amplitude_param + phase_param == strength, staying valid
        # for the FULL strength range [0, 1]
        error_1q = phase_amplitude_damping_error(strength/2, strength/2)
        error_2q = error_1q.tensor(error_1q)

        noise_model.add_all_qubit_quantum_error(error_1q, one_qubit_gates)
        noise_model.add_all_qubit_quantum_error(error_2q, two_qubit_gates)

    elif noise_type == "thermal_relaxation":
        # Simple parameterization.
        # Gate time is fixed and T1/T2 are controlled indirectly by strength.
        gate_time = 100
        t1 = max(1.0, 1000 / (strength + 1e-6))
        t2 = max(1.0, 800 / (strength + 1e-6))

        error_1q = thermal_relaxation_error(t1, t2, gate_time)
        error_2q = error_1q.tensor(error_1q)

        noise_model.add_all_qubit_quantum_error(error_1q, one_qubit_gates)
        noise_model.add_all_qubit_quantum_error(error_2q, two_qubit_gates)

    elif noise_type == "bit_flip":
        error_1q = pauli_error([
            ("X", strength),
            ("I", 1 - strength),
        ])
        error_2q = error_1q.tensor(error_1q)

        noise_model.add_all_qubit_quantum_error(error_1q, one_qubit_gates)
        noise_model.add_all_qubit_quantum_error(error_2q, two_qubit_gates)

    elif noise_type == "phase_flip":
        error_1q = pauli_error([
            ("Z", strength),
            ("I", 1 - strength),
        ])
        error_2q = error_1q.tensor(error_1q)

        noise_model.add_all_qubit_quantum_error(error_1q, one_qubit_gates)
        noise_model.add_all_qubit_quantum_error(error_2q, two_qubit_gates)

    elif noise_type == "pauli_asymmetric":
        # Asymmetric Pauli noise.
        # This is intentionally different from depolarizing noise.
        px = 0.60 * strength
        py = 0.25 * strength
        pz = 0.15 * strength
        pi = 1.0 - strength

        error_1q = pauli_error([
            ("X", px),
            ("Y", py),
            ("Z", pz),
            ("I", pi),
        ])
        error_2q = error_1q.tensor(error_1q)

        noise_model.add_all_qubit_quantum_error(error_1q, one_qubit_gates)
        noise_model.add_all_qubit_quantum_error(error_2q, two_qubit_gates)

    elif noise_type == "readout":
        # Symmetric readout confusion.
        p = strength
        readout_error = ReadoutError([
            [1 - p, p],
            [p, 1 - p],
        ])

        noise_model.add_all_qubit_readout_error(readout_error)

    elif noise_type == "reset":
        error_1q = reset_error(strength)
        error_2q = error_1q.tensor(error_1q)

        noise_model.add_all_qubit_quantum_error(error_1q, one_qubit_gates)
        noise_model.add_all_qubit_quantum_error(error_2q, two_qubit_gates)

    return noise_model