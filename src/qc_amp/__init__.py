"""
QC-Amp: Quantum Computing for QCD Amplitude Calculations

A Qiskit-based library for computing colour factors in QCD amplitudes
using quantum circuits.

This package implements the quantum algorithms described in:

    H. A. Chawdhry and M. Pellen,
    "Quantum simulation of colour in perturbative quantum chromodynamics",
    SciPost Phys. 15, 205 (2023). arXiv:2303.04818

Modules:
    su3: SU(3) group theory utilities (Gell-Mann matrices, structure constants)
    gates: Quantum gate definitions (A, B, Lambda, M, Q, G gates)
    circuits: Circuit builders for state preparation and full diagrams
    colour_factors: Colour factor computation from quantum circuits

Example:
    >>> from qc_amp import compute_colour_factor
    >>> from qc_amp.circuits import quark_emission_absorption
    >>> 
    >>> circuit = quark_emission_absorption()
    >>> C = compute_colour_factor(circuit, n_quarks=1, n_gluons=1)
    >>> print(f"Colour factor: {C}")
"""

from qc_amp.su3 import (
    GELL_MANN_MATRICES,
    UNITARY_ADJUSTED_MATRICES,
    su3_structure_constants,
    expand_matrix,
)
from qc_amp.gates import (
    R_MATRIX,
    R_GATE,
    A_gate,
    B1_gate,
    B_gate,
    mu_coefficient,
    Lambda_gate,
    M_gate,
    Q_gate,
    G_prime_gate,
    G_gate,
)
from qc_amp.circuits import (
    R_quark_prep,
    R_gluon_prep,
    quark_emission_absorption,
)
from qc_amp.colour_factors import compute_colour_factor

__version__ = "0.1.0"
__author__ = "QC-Amp Contributors"

__all__ = [
    # SU(3) utilities
    "GELL_MANN_MATRICES",
    "UNITARY_ADJUSTED_MATRICES", 
    "su3_structure_constants",
    "expand_matrix",
    # Gates
    "R_MATRIX",
    "R_GATE",
    "A_gate",
    "B1_gate",
    "B_gate",
    "mu_coefficient",
    "Lambda_gate",
    "M_gate",
    "Q_gate",
    "G_prime_gate",
    "G_gate",
    # Circuits
    "R_quark_prep",
    "R_gluon_prep",
    "quark_emission_absorption",
    # Colour factors
    "compute_colour_factor",
]
