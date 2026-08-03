"""
circuits.py

Probe circuit construction for shadow-based quantum noise fingerprinting.

This module defines the probe circuits used to generate measurement data
for noise classification. Two complementary families of probes are used:

1. Simple structured probes: computational basis states (|00...0>,
   |11...1>), uniform superposition (|++...+>), and a fully-entangled
   GHZ state. Each probe is sensitive to a different aspect of the
   noise channel: the basis states are primarily sensitive to
   population-changing errors (e.g. bit flips, amplitude damping), the
   superposition state is primarily sensitive to coherence-destroying
   errors (e.g. phase flips, dephasing), and the GHZ state probes correlations
   across all qubits, making it sensitive to noise that disrupts entanglement 
   and its entangling structure scales with n_qubits.

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


def build_ghz_circuit(n_qubits: int = 2) -> QuantumCircuit:
    """
    Construct a fully-entangled GHZ state across ALL qubits:
        |GHZ> = (|00...0> + |11...1>) / sqrt(2)

    Unlike a fixed-pair Bell state, this probe's entangling structure
    scales with n_qubits -- every qubit participates in the
    entanglement, rather than leaving extra qubits idle. At
    n_qubits=2 this is exactly the standard Bell state.

    Construction: H on qubit 0, then a chain of CX gates
    (0->1, 1->2, ..., n-2->n-1) to propagate the entanglement across
    every qubit.

    Args:
        n_qubits: Number of qubits in the circuit. Must be at least 2.

    Returns:
        A QuantumCircuit with a GHZ state prepared across all qubits.

    Raises:
        ValueError: If n_qubits is less than 2.
    """
    if n_qubits < 2:
        raise ValueError("GHZ circuit requires at least 2 qubits.")

    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for q in range(n_qubits - 1):
        qc.cx(q, q + 1)
    return qc


# Backward-compatible alias -- old code calling build_bell_circuit
# still works, but now gets the fully-entangling GHZ construction
# rather than the original fixed-pair Bell state.
def build_bell_circuit(n_qubits: int = 2) -> QuantumCircuit:
    """
    Alias for build_ghz_circuit, kept for backward compatibility with
    any code still calling build_bell_circuit by name. See
    build_ghz_circuit's docstring -- at n_qubits=2 this is identical
    to the original Bell-state behavior; for n_qubits > 2 it now
    entangles ALL qubits instead of leaving extras idle.
    """
    return build_ghz_circuit(n_qubits)


def build_qaoa_circuit(
    gamma: float,
    beta: float,
    n_qubits: int = 2,
) -> QuantumCircuit:
    """
    Build a small QAOA-like probe circuit.

    This is not meant to solve an optimization problem.
    It is used as a structured, parameterized probe circuit.

    For 2 qubits:
        Start in |++>
        Apply ZZ-type phase interaction
        Apply X rotations
    """

    qc = QuantumCircuit(n_qubits)

    # Initial uniform superposition
    for q in range(n_qubits):
        qc.h(q)

    # Simple ZZ cost layer using CX-RZ-CX pattern
    for q in range(n_qubits - 1):
        qc.cx(q, q + 1)
        qc.rz(2 * gamma, q + 1)
        qc.cx(q, q + 1)

    # Mixer layer
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
    Return a list of probe circuits.

    Args:
        n_qubits:
            Number of qubits.
        num_qaoa_probes:
            Number of random QAOA-like circuits.
        seed:
            Random seed.
        include_simple_probes:
            Whether to include clean basis/superposition/GHZ probes.

    Returns:
        List of tuples:
            [(probe_name, circuit), ...]
    """

    probes = []

    if include_simple_probes:
        probes.append(("zero", build_basis_zero_circuit(n_qubits)))
        probes.append(("one", build_basis_one_circuit(n_qubits)))
        probes.append(("plus", build_plus_circuit(n_qubits)))

        if n_qubits >= 2:
            probes.append(("ghz", build_ghz_circuit(n_qubits)))

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