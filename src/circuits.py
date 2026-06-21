"""
circuits.py

Probe circuit construction for shadow-based quantum noise fingerprinting.

This module defines the probe circuits used to generate measurement data
for noise classification. Two complementary families of probes are used:

1. Simple structured probes: computational basis states (|00...0>,
   |11...1>), uniform superposition (|++...+>), and a Bell state. Each
   probe is sensitive to a different aspect of the noise channel: the
   basis states are primarily sensitive to population-changing errors
   (e.g. bit flips, amplitude damping), the superposition state is
   primarily sensitive to coherence-destroying errors (e.g. phase
   flips, dephasing), and the Bell state additionally probes
   correlations between qubits, making it sensitive to noise that
   disrupts entanglement.

2. QAOA-style probes: shallow circuits that follow the structural
   template of the Quantum Approximate Optimization Algorithm (QAOA),
   with one entangling/cost layer followed by one mixer layer. These
   circuits are not used to solve an optimization problem; the QAOA
   structure is used purely to generate richer, entangled probe states
   than the simple probes provide. Diversity across probe instances is
   achieved by randomly sampling the layer parameters (gamma, beta)
   for each circuit, rather than by varying the circuit structure
   itself.

Combining both families ensures the resulting feature space captures
population, coherence, and entanglement signatures of the underlying
noise model.
"""

import numpy as np
from qiskit import QuantumCircuit


def build_basis_zero_circuit(n_qubits: int = 2) -> QuantumCircuit:
    """
    Construct the computational basis state |00...0>.

    Args:
        n_qubits: Number of qubits in the circuit.

    Returns:
        A QuantumCircuit prepared in the all-zero basis state.
    """
    qc = QuantumCircuit(n_qubits)
    return qc


def build_basis_one_circuit(n_qubits: int = 2) -> QuantumCircuit:
    """
    Construct the computational basis state |11...1>.

    Args:
        n_qubits: Number of qubits in the circuit.

    Returns:
        A QuantumCircuit prepared in the all-one basis state.
    """
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.x(q)
    return qc


def build_plus_circuit(n_qubits: int = 2) -> QuantumCircuit:
    """
    Construct the uniform superposition state |++...+>.

    This probe is primarily sensitive to coherence-destroying noise
    channels, since it places all qubits in an equal superposition
    of |0> and |1>.

    Args:
        n_qubits: Number of qubits in the circuit.

    Returns:
        A QuantumCircuit prepared in the uniform superposition state.
    """
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)
    return qc


def build_bell_circuit(n_qubits: int = 2) -> QuantumCircuit:
    """
    Construct a Bell-state probe on the first two qubits.

    For n_qubits > 2, any additional qubits remain in |0> and are not
    entangled with the Bell pair. This probe introduces entanglement,
    making it sensitive to noise that disrupts correlations between
    qubits.

    Args:
        n_qubits: Number of qubits in the circuit. Must be at least 2.

    Returns:
        A QuantumCircuit with a Bell state prepared on qubits 0 and 1.

    Raises:
        ValueError: If n_qubits is less than 2.
    """
    if n_qubits < 2:
        raise ValueError("Bell circuit requires at least 2 qubits.")

    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def build_qaoa_circuit(
    gamma: float,
    beta: float,
    n_qubits: int = 2,
) -> QuantumCircuit:
    """
    Construct a shallow QAOA-style probe circuit.

    This circuit is not used to solve an optimization problem. It is
    used purely as a structured, parameterized probe state that
    combines entanglement with tunable single-qubit rotations,
    providing richer circuit behavior than the simple probes.

    Circuit structure (single-layer ansatz):
        1. Initialize all qubits in a uniform superposition.
        2. Apply one ZZ-type entangling (cost) layer between
           neighboring qubits, parameterized by gamma.
        3. Apply one X-rotation mixer layer to all qubits,
           parameterized by beta.

    Args:
        gamma: Rotation angle for the entangling (cost) layer.
        beta: Rotation angle for the mixer layer.
        n_qubits: Number of qubits in the circuit.

    Returns:
        A parameterized QuantumCircuit implementing the QAOA-style probe.
    """

    qc = QuantumCircuit(n_qubits)

    # Initialize all qubits in a uniform superposition
    for q in range(n_qubits):
        qc.h(q)

    # Entangling layer: ZZ-type interaction via CX-RZ-CX decomposition
    for q in range(n_qubits - 1):
        qc.cx(q, q + 1)
        qc.rz(2 * gamma, q + 1)
        qc.cx(q, q + 1)

    # Mixer layer: single-qubit X-rotations
    for q in range(n_qubits):
        qc.rx(2 * beta, q)

    return qc


def get_probe_circuits(
    n_qubits: int = 2,
    num_qaoa_probes: int = 5,
    seed: int = 42,
    include_simple_probes: bool = True,
):
    """
    Generate the full set of probe circuits used for noise fingerprinting.

    Args:
        n_qubits: Number of qubits used in each probe circuit.
        num_qaoa_probes: Number of randomly parameterized QAOA-style
            probe circuits to generate.
        seed: Random seed used to sample QAOA circuit parameters,
            ensuring reproducibility across runs.
        include_simple_probes: Whether to include the simple structured
            probes (basis states, superposition, and Bell state) in
            addition to the QAOA-style probes.

    Returns:
        A list of (probe_name, circuit) tuples, where probe_name is a
        string identifier and circuit is the corresponding QuantumCircuit.
    """

    probes = []

    if include_simple_probes:
        probes.append(("zero", build_basis_zero_circuit(n_qubits)))
        probes.append(("one", build_basis_one_circuit(n_qubits)))
        probes.append(("plus", build_plus_circuit(n_qubits)))

        if n_qubits >= 2:
            probes.append(("bell", build_bell_circuit(n_qubits)))

    rng = np.random.default_rng(seed)

    for i in range(num_qaoa_probes):
        gamma = rng.uniform(0, np.pi)
        beta = rng.uniform(0, np.pi)

        qc = build_qaoa_circuit(
            gamma=gamma,
            beta=beta,
            n_qubits=n_qubits,
        )

        probes.append((f"qaoa_{i}", qc))

    return probes