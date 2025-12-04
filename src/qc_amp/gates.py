"""
Quantum Gate Definitions for QCD Amplitude Calculations

This module provides the quantum gates used in colour factor calculations,
as described in Chawdhry & Pellen, SciPost Phys. 15, 205 (2023).

Unitarisation Gates:
    - A: Increment operator on unitarisation register (Eq. 25)
    - B1: Single-qubit rotation gate (Eq. 26)
    - B: Controlled B1 gate on unitarisation register

Colour Interaction Gates:
    - Lambda (Λ): Gluon-controlled colour rotation on quark (Eq. 29)
    - M: Unitarisation correction operator (Eq. 31)
    - Q: Complete quark-gluon interaction vertex (Eq. 30)

Triple-Gluon Gates:
    - G_prime (G'): Structure-constant weighted rotations (Eq. 36)
    - G: Complete triple-gluon vertex (Eq. 35)

State Preparation:
    - R_MATRIX: 4×4 unitary for quark singlet preparation (Appendix A)
    - R_GATE: Qiskit UnitaryGate wrapping R_MATRIX
"""

from typing import List
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

from qc_amp.su3 import (
    GELL_MANN_MATRICES,
    UNITARY_ADJUSTED_MATRICES,
    expand_matrix,
    get_structure_constants,
)

__all__ = [
    "R_MATRIX",
    "R_GATE",
    "A_gate",
    "B1_gate",
    "B_gate",
    "mu_coefficient",
    "quark_colour_bits",
    "Lambda_gate",
    "M_gate",
    "Q_gate",
    "G_prime_gate",
    "G_gate",
]


# =============================================================================
# R Gate: Quark Singlet Preparation
# =============================================================================

R_MATRIX = np.array([
    [np.sqrt(1/3),  np.sqrt(1/2), -np.sqrt(1/6), 0],
    [np.sqrt(1/3), -np.sqrt(1/2), -np.sqrt(1/6), 0],
    [np.sqrt(1/3),  0,             np.sqrt(2/3), 0],
    [0,             0,             0,            1],
], dtype=complex)
"""
4×4 R matrix for quark qutrit state preparation.

Maps |00⟩ → (1/√3)(|00⟩ + |01⟩ + |10⟩), an equal superposition
over the three colour states. The |11⟩ state is left invariant.
"""

R_GATE = UnitaryGate(R_MATRIX, label="R")
"""Qiskit UnitaryGate wrapping R_MATRIX."""


# =============================================================================
# Unitarisation Register Gates: A, B1, B
# =============================================================================

def A_gate(register: QuantumRegister) -> QuantumCircuit:
    """
    Increment operator A on the unitarisation register.
    
    Implements the cyclic increment:
        A|k⟩ = |k+1 mod 2^N⟩
    
    where N is the number of qubits in the register.
    register[0] is the least significant bit.
    
    Args:
        register: QuantumRegister for the unitarisation qubits.
        
    Returns:
        QuantumCircuit implementing the increment operation.
        
    Example:
        >>> U = QuantumRegister(3, 'U')
        >>> A_circ = A_gate(U)
        >>> A_circ.draw()
    """
    n = len(register)
    qc = QuantumCircuit(register, name="A")
    
    # Multi-controlled X gates for ripple-carry increment
    for i in range(n - 1):
        controls = [register[j] for j in range(n - i - 1)]
        target = register[n - i - 1]
        qc.mcx(controls, target)
    
    qc.x(register[0])
    return qc


def B1_gate(alpha: complex) -> UnitaryGate:
    """
    Single-qubit B1(α) rotation gate.
    
    Implements the 2×2 unitary:
        B1(α) = [[√(1-|α|²),  α    ],
                 [   -α*   , √(1-|α|²)]]
    
    This is used to encode amplitude coefficients into the
    unitarisation register.
    
    Args:
        alpha: Complex coefficient with |α|² ≤ 1.
        
    Returns:
        UnitaryGate implementing B1(α).
        
    Raises:
        ValueError: If |α|² > 1.
        
    Example:
        >>> gate = B1_gate(0.5)
        >>> Operator(gate).is_unitary()
        True
    """
    a = complex(alpha)
    if abs(a)**2 > 1 + 1e-12:
        raise ValueError(f"|alpha|² must be ≤ 1, got |{alpha}|² = {abs(a)**2}")
    
    s = np.sqrt(1 - abs(a)**2)
    mat = np.array([
        [s, a],
        [-np.conj(a), s]
    ], dtype=complex)
    
    return UnitaryGate(mat, label=f"B1({alpha:.3f})")


def B_gate(alpha: complex, U: QuantumRegister) -> QuantumCircuit:
    """
    B(α) gate on the unitarisation register.
    
    Applies B1(α) to U[0], controlled on U[1:] all being in state |0⟩.
    This ensures the rotation only affects the "unused" portion of
    the unitarisation space.
    
    Args:
        alpha: Complex coefficient with |α|² ≤ 1.
        U: QuantumRegister for unitarisation qubits.
        
    Returns:
        QuantumCircuit implementing controlled B1(α).
        
    Example:
        >>> U = QuantumRegister(3, 'U')
        >>> B_circ = B_gate(0.5, U)
    """
    NU = len(U)
    qc = QuantumCircuit(U, name=f"B({alpha:.3f})")
    
    if NU == 1:
        qc.append(B1_gate(alpha), [U[0]])
        return qc
    
    base = B1_gate(alpha)
    ctrl_state = "0" * (NU - 1)
    controlled = base.control(num_ctrl_qubits=NU - 1, ctrl_state=ctrl_state)
    qc.append(controlled, list(U[1:]) + [U[0]])
    
    return qc


# =============================================================================
# μ Coefficient
# =============================================================================

def mu_coefficient(a: int, i: int) -> complex:
    """
    Compute μ(a,i) coefficient for the M operator.
    
    This coefficient determines the amplitude contribution from
    gluon colour a and quark colour i in the unitarisation procedure.
    
    The coefficient is:
        - 1/2 if row i of λ̂_a equals row i of λ_a
        - Special values for a=8 (the diagonal generator)
        - 0 otherwise
    
    Args:
        a: Gluon colour index (1 to 8).
        i: Quark colour index (1 to 3).
        
    Returns:
        The μ(a,i) coefficient as a complex number.
        
    Example:
        >>> mu_coefficient(1, 1)  # λ̂_1 row 1 differs from λ_1
        0.0
        >>> mu_coefficient(3, 1)  # λ̂_3 row 1 equals λ_3 row 1
        0.5
    """
    l_unit = UNITARY_ADJUSTED_MATRICES[a - 1]
    l_orig = GELL_MANN_MATRICES[a - 1]
    
    # μ(a,i) encodes the correction needed when λ̂_a differs from λ_a
    # From Eq. (28) and (31) in the paper:
    # - When row i of λ̂_a equals row i of λ_a: no correction, μ = 0
    # - When they differ: μ = (λ_a)_{ii} / 2 (the original amplitude factor)
    
    # Check if the i-th row matches
    row_equal = np.allclose(l_unit[i - 1, :], l_orig[i - 1, :], atol=1e-10)
    
    if row_equal:
        # Row matches: the unitary λ̂_a already gives correct action
        # The amplitude factor is 1/2 (from T^a = λ_a/2 normalization)
        return 0.5
    
    # Row differs: need to account for what λ_a would have contributed
    # For off-diagonal Gell-Mann matrices acting on "wrong" rows: contribution is 0
    # For diagonal ones (λ_3, λ_8): contribution is λ_a,ii / 2
    
    # Get the diagonal element that should have been applied
    orig_diag = l_orig[i - 1, i - 1]
    
    # Return half of the original diagonal (T = λ/2 normalization)
    return complex(orig_diag) / 2


def quark_colour_bits(i: int, n_qubits: int = 2) -> str:
    """
    Map quark colour index to qubit basis state string.
    
    Encoding:
        colour 1 (red)   → |00⟩
        colour 2 (green) → |01⟩
        colour 3 (blue)  → |10⟩
        (|11⟩ is unused)
    
    Args:
        i: Quark colour index (1, 2, or 3).
        n_qubits: Number of qubits (must be 2 for this encoding).
        
    Returns:
        Binary string representing the basis state.
        
    Raises:
        ValueError: If i not in {1, 2, 3} or n_qubits ≠ 2.
    """
    if n_qubits != 2:
        raise ValueError("This mapping assumes 2 qubits for the quark register.")
    
    mapping = {1: "00", 2: "01", 3: "10"}
    if i not in mapping:
        raise ValueError(f"Quark colour index must be 1, 2, or 3, got {i}")
    
    return mapping[i]


# =============================================================================
# Colour Interaction Gates: Lambda, M, Q
# =============================================================================

def Lambda_gate(quark: QuantumRegister, gluon: QuantumRegister) -> QuantumCircuit:
    """
    Λ (Lambda) gate: gluon-controlled colour rotation on quark.
    
    Implements:
        Λ = ∏_a C_{|a⟩_g}[λ̂_a]
    
    For each gluon colour state |a⟩, applies the corresponding
    unitary-adjusted Gell-Mann matrix λ̂_a to the quark register.
    
    Args:
        quark: QuantumRegister for quark colour (2 qubits).
        gluon: QuantumRegister for gluon colour (3 qubits).
        
    Returns:
        QuantumCircuit implementing the Λ gate.
        
    Example:
        >>> q = QuantumRegister(2, 'q')
        >>> g = QuantumRegister(3, 'g')
        >>> Lambda_circ = Lambda_gate(q, g)
    """
    qc = QuantumCircuit(quark, gluon, name="Λ")
    n_ctrl = len(gluon)
    
    for a, L in enumerate(UNITARY_ADJUSTED_MATRICES, start=1):
        expanded = expand_matrix(L)
        assert Operator(expanded).is_unitary(), f"Matrix λ̂{a} is not unitary after expansion."
        
        base_gate = UnitaryGate(expanded, label=f"λ̂{a}")
        
        # Control on gluon state |a-1⟩ (0-indexed binary)
        ctrl_state = format(a - 1, f"0{n_ctrl}b")
        CU = base_gate.control(num_ctrl_qubits=n_ctrl, ctrl_state=ctrl_state)
        
        # Controls (gluon) first, then targets (quark)
        qc.append(CU, gluon[:] + quark[:])
    
    return qc


def M_gate(
    gluon: QuantumRegister,
    quark: QuantumRegister,
    U: QuantumRegister
) -> QuantumCircuit:
    """
    M operator for unitarisation correction.
    
    Implements:
        M = ∏_{a,i : μ(a,i)≠0} C_{|a⟩_g |i⟩_q}[B(μ(a,i))]
    
    For each (gluon colour a, quark colour i) pair with non-zero
    μ coefficient, applies the controlled B rotation to the
    unitarisation register.
    
    Args:
        gluon: QuantumRegister for gluon colour (3 qubits).
        quark: QuantumRegister for quark colour (2 qubits).
        U: QuantumRegister for unitarisation (variable size).
        
    Returns:
        QuantumCircuit implementing the M operator.
    """
    qc = QuantumCircuit(gluon, quark, U, name="M")
    n_g = len(gluon)
    n_q = len(quark)
    
    for a in range(1, 9):  # gluon colours 1-8
        for i in range(1, 4):  # quark colours 1-3
            alpha = mu_coefficient(a, i)
            
            if abs(alpha) < 1e-12:
                continue
            
            # Build B(μ(a,i)) on U and control on (g, q)
            B_circ = B_gate(alpha, U)
            B_gate_obj = B_circ.to_gate(label=f"B({alpha:.3f})")
            
            # Control qubits are ordered [g0, g1, g2, q0, q1]
            # State number = gluon_state + 8 * quark_state
            # where gluon_state = a-1 (0-7) and quark_state = i-1 (0-2)
            gluon_state = a - 1
            quark_state = i - 1  # maps colour 1,2,3 to state 0,1,2
            ctrl_state_int = gluon_state + (2**n_g) * quark_state
            ctrl_state = format(ctrl_state_int, f"0{n_g + n_q}b")
            
            controlled = B_gate_obj.control(
                num_ctrl_qubits=n_g + n_q,
                ctrl_state=ctrl_state
            )
            qc.append(controlled, list(gluon) + list(quark) + list(U))
    
    return qc


def Q_gate(
    gluon: QuantumRegister,
    quark: QuantumRegister,
    U: QuantumRegister
) -> QuantumCircuit:
    """
    Q gate: complete quark-gluon interaction vertex.
    
    Implements:
        Q = (Λ ⊗ 1_U) · M · (1_g ⊗ 1_q ⊗ A)
    
    This represents a single quark-gluon vertex in a Feynman diagram,
    including the unitarisation machinery.
    
    Args:
        gluon: QuantumRegister for gluon colour (3 qubits).
        quark: QuantumRegister for quark colour (2 qubits).
        U: QuantumRegister for unitarisation.
        
    Returns:
        QuantumCircuit implementing the Q gate.
        
    Example:
        >>> g = QuantumRegister(3, 'g')
        >>> q = QuantumRegister(2, 'q')
        >>> U = QuantumRegister(3, 'U')
        >>> Q_circ = Q_gate(g, q, U)
    """
    qc = QuantumCircuit(gluon, quark, U, name="Q")
    
    # 1. Increment U with A
    qc.compose(A_gate(U), qubits=U, inplace=True)
    
    # 2. Conditional correction via M
    qc.compose(
        M_gate(gluon, quark, U),
        qubits=list(gluon) + list(quark) + list(U),
        inplace=True
    )
    
    # 3. Apply Λ on (quark, gluon)
    qc.compose(
        Lambda_gate(quark, gluon),
        qubits=list(quark) + list(gluon),
        inplace=True
    )
    
    return qc


# =============================================================================
# Triple-Gluon Gates: G', G
# =============================================================================

def G_prime_gate(
    g1: QuantumRegister,
    g2: QuantumRegister,
    g3: QuantumRegister,
    U: QuantumRegister
) -> QuantumCircuit:
    """
    G' gate: structure-constant weighted rotations for triple-gluon vertex.
    
    Implements:
        G' = ∏_{a,b,c : f_abc≠0} C_{|a⟩|b⟩|c⟩}[B(f_abc)]
    
    For each non-zero structure constant f_abc, applies a controlled
    B rotation weighted by f_abc.
    
    Args:
        g1, g2, g3: QuantumRegisters for the three gluon colours (3 qubits each).
        U: QuantumRegister for unitarisation.
        
    Returns:
        QuantumCircuit implementing G'.
    """
    qc = QuantumCircuit(g1, g2, g3, U, name="G'")
    
    n_g = len(g1)
    assert len(g2) == n_g == len(g3), "All gluon registers must have same size"
    
    F_abc = get_structure_constants()
    
    for (a, b, c), val in F_abc.items():
        alpha = val
        
        # Build B(f_abc) on U
        B_circ = B_gate(alpha, U)
        B_gate_obj = B_circ.to_gate(label=f"B({alpha:.3f})")
        
        # Controls: g1|a⟩, g2|b⟩, g3|c⟩
        bits_a = format(a - 1, f"0{n_g}b")
        bits_b = format(b - 1, f"0{n_g}b")
        bits_c = format(c - 1, f"0{n_g}b")
        ctrl_state = bits_a + bits_b + bits_c
        
        controlled = B_gate_obj.control(
            num_ctrl_qubits=3 * n_g,
            ctrl_state=ctrl_state
        )
        qc.append(controlled, list(g1) + list(g2) + list(g3) + list(U))
    
    return qc


def G_gate(
    g1: QuantumRegister,
    g2: QuantumRegister,
    g3: QuantumRegister,
    U: QuantumRegister
) -> QuantumCircuit:
    """
    G gate: complete triple-gluon vertex.
    
    Implements:
        G = G' · A
    
    This represents a triple-gluon vertex in a Feynman diagram.
    
    Args:
        g1, g2, g3: QuantumRegisters for the three gluon colours.
        U: QuantumRegister for unitarisation.
        
    Returns:
        QuantumCircuit implementing the G gate.
    """
    qc = QuantumCircuit(g1, g2, g3, U, name="G")
    
    # 1. Increment U
    qc.compose(A_gate(U), qubits=U, inplace=True)
    
    # 2. Apply G'
    qc.compose(
        G_prime_gate(g1, g2, g3, U),
        qubits=list(g1) + list(g2) + list(g3) + list(U),
        inplace=True
    )
    
    return qc
