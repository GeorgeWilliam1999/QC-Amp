"""
Colour Factor Computation

This module provides functions to compute QCD colour factors
from quantum circuits, as described in Chawdhry & Pellen,
SciPost Phys. 15, 205 (2023).

The colour factor C is computed as (Eq. 37):
    C = N × ⟨Ω_all|ψ_final⟩

where:
    - N = N_c^{n_q} × (N_c² - 1)^{n_g} is the normalization factor
    - n_q = number of quark lines
    - n_g = number of gluon lines  
    - N_c = 3 for SU(3)
    - |Ω_all⟩ = |0...0⟩ is the reference state
    - |ψ_final⟩ is the state after evolving through the circuit
"""

from typing import Tuple, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

__all__ = [
    "compute_colour_factor",
    "compute_colour_factor_detailed",
]


def compute_colour_factor(
    circuit: QuantumCircuit,
    n_quarks: int = 1,
    n_gluons: int = 1,
    N_c: int = 3
) -> complex:
    """
    Compute the colour factor from a quantum circuit.
    
    The colour factor is extracted from the amplitude of returning
    to the initial |0...0⟩ state after the full circuit evolution.
    
    Args:
        circuit: QuantumCircuit implementing the Feynman diagram
                with preparation gates, interaction vertices,
                and inverse preparation gates.
        n_quarks: Number of quark lines in the diagram.
        n_gluons: Number of gluon lines in the diagram.
        N_c: Number of colours (3 for QCD).
        
    Returns:
        The colour factor C as a complex number.
        For physical diagrams, this should be real.
        
    Example:
        >>> from qc_amp.circuits import quark_emission_absorption
        >>> circ = quark_emission_absorption()
        >>> C = compute_colour_factor(circ, n_quarks=1, n_gluons=1)
        >>> print(f"Colour factor: {C.real:.4f}")
        Colour factor: 4.0000
    """
    C, _, _ = compute_colour_factor_detailed(circuit, n_quarks, n_gluons, N_c)
    return C


def compute_colour_factor_detailed(
    circuit: QuantumCircuit,
    n_quarks: int = 1,
    n_gluons: int = 1,
    N_c: int = 3
) -> Tuple[complex, complex, int]:
    """
    Compute the colour factor with detailed intermediate results.
    
    Args:
        circuit: QuantumCircuit implementing the Feynman diagram.
        n_quarks: Number of quark lines.
        n_gluons: Number of gluon lines.
        N_c: Number of colours (3 for QCD).
        
    Returns:
        Tuple of (C, amplitude, N) where:
            - C: The colour factor
            - amplitude: The raw ⟨0...0|ψ_final⟩ amplitude
            - N: The normalization factor
            
    Example:
        >>> from qc_amp.circuits import quark_emission_absorption
        >>> circ = quark_emission_absorption()
        >>> C, amp, N = compute_colour_factor_detailed(circ)
        >>> print(f"N = {N}, amplitude = {amp:.6f}, C = {C:.4f}")
    """
    # 1. Prepare initial state |0...0⟩
    psi0 = Statevector.from_label("0" * circuit.num_qubits)
    
    # 2. Evolve through the circuit
    psi_final = psi0.evolve(circuit)
    
    # 3. Extract amplitude of |0...0⟩ (first component in computational basis)
    amp_omega = psi_final.data[0]
    
    # 4. Compute normalization factor
    # N = N_c^{n_q} × (N_c² - 1)^{n_g}
    N = (N_c ** n_quarks) * ((N_c**2 - 1) ** n_gluons)
    
    # 5. Colour factor
    C = N * amp_omega
    
    return C, amp_omega, N


def verify_colour_factor(
    computed: complex,
    expected: float,
    tolerance: float = 1e-6
) -> bool:
    """
    Verify a computed colour factor against an expected value.
    
    Args:
        computed: The computed colour factor.
        expected: The expected (analytic) value.
        tolerance: Relative tolerance for comparison.
        
    Returns:
        True if |computed - expected| / |expected| < tolerance.
        
    Example:
        >>> C = compute_colour_factor(circ)
        >>> verify_colour_factor(C, expected=4.0)
        True
    """
    if abs(expected) < 1e-12:
        return abs(computed) < tolerance
    
    relative_error = abs(computed - expected) / abs(expected)
    return relative_error < tolerance


def format_colour_factor_result(
    circuit: QuantumCircuit,
    n_quarks: int = 1,
    n_gluons: int = 1,
    expected: Optional[float] = None,
    N_c: int = 3
) -> str:
    """
    Format colour factor computation results for display.
    
    Args:
        circuit: The quantum circuit.
        n_quarks: Number of quark lines.
        n_gluons: Number of gluon lines.
        expected: Optional expected value for comparison.
        N_c: Number of colours.
        
    Returns:
        Formatted string with computation results.
    """
    C, amp, N = compute_colour_factor_detailed(circuit, n_quarks, n_gluons, N_c)
    
    lines = [
        f"Colour Factor Computation Results",
        f"=" * 40,
        f"Normalization N = {N}",
        f"Amplitude ⟨Ω|ψ⟩ = {amp:.6f}",
        f"Colour factor C = N × amplitude = {C:.6f}",
        f"|C| (magnitude) = {abs(C):.6f}",
    ]
    
    if expected is not None:
        rel_error = abs(C - expected) / abs(expected) if abs(expected) > 1e-12 else abs(C)
        lines.extend([
            f"",
            f"Expected C = {expected}",
            f"Relative error = {rel_error:.2e}",
        ])
    
    return "\n".join(lines)
