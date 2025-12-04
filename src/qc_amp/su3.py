"""
SU(3) Group Theory Utilities

This module provides the mathematical foundations for SU(3) colour calculations:
- Gell-Mann matrices (generators of SU(3))
- Unitary-adjusted matrices for quantum circuit implementation
- Structure constants f_abc
- Helper functions for matrix embedding

References:
    Chawdhry & Pellen, SciPost Phys. 15, 205 (2023), arXiv:2303.04818
    
    - Gell-Mann matrices λ_a: Eq. (3)
    - Unitary-adjusted matrices λ̂_a: Eq. (28)
    - Structure constants f_abc: Eq. (35)
"""

from typing import Dict, List, Tuple
import numpy as np

__all__ = [
    "GELL_MANN_MATRICES",
    "UNITARY_ADJUSTED_MATRICES",
    "su3_structure_constants",
    "expand_matrix",
    "is_unitary",
]


# =============================================================================
# Gell-Mann Matrices λ_a (a = 1, ..., 8)
# =============================================================================
# These are the standard generators of SU(3), satisfying:
#   - Hermitian: λ_a† = λ_a
#   - Traceless: Tr(λ_a) = 0
#   - Normalization: Tr(λ_a λ_b) = 2δ_ab

L1 = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 0]
], dtype=complex)

L2 = np.array([
    [0, -1j, 0],
    [1j, 0, 0],
    [0, 0, 0]
], dtype=complex)

L3 = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, 0]
], dtype=complex)

L4 = np.array([
    [0, 0, 1],
    [0, 0, 0],
    [1, 0, 0]
], dtype=complex)

L5 = np.array([
    [0, 0, -1j],
    [0, 0, 0],
    [1j, 0, 0]
], dtype=complex)

L6 = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
], dtype=complex)

L7 = np.array([
    [0, 0, 0],
    [0, 0, -1j],
    [0, 1j, 0]
], dtype=complex)

L8 = (1 / np.sqrt(3)) * np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, -2]
], dtype=complex)

GELL_MANN_MATRICES: List[np.ndarray] = [L1, L2, L3, L4, L5, L6, L7, L8]
"""List of 8 Gell-Mann matrices λ_1 through λ_8."""


# =============================================================================
# Unitary-Adjusted Matrices λ̂_a
# =============================================================================
# Modified versions of Gell-Mann matrices that are unitary (not just Hermitian).
# Required for direct implementation as quantum gates.
# 
# The adjustment involves completing the matrices to be unitary by adding
# identity elements where the original Gell-Mann matrices have zeros.

l1 = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]
], dtype=complex)

l2 = np.array([
    [0, -1j, 0],
    [1j, 0, 0],
    [0, 0, 1]
], dtype=complex)

l3 = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
], dtype=complex)

l4 = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0]
], dtype=complex)

l5 = np.array([
    [0, 0, -1j],
    [0, 1, 0],
    [1j, 0, 0]
], dtype=complex)

l6 = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
], dtype=complex)

l7 = np.array([
    [1, 0, 0],
    [0, 0, -1j],
    [0, 1j, 0]
], dtype=complex)

# λ̂_8: The Gell-Mann matrix λ_8 is diagonal: diag(1, 1, -2)/√3
# This is NOT unitary (eigenvalues are not on unit circle).
# The paper (Eq. 28) defines λ̂_a to be unitary matrices that match λ_a
# on certain rows (where the action matters for colour amplitudes).
# 
# For λ_8, since all diagonal elements are different and the matrix doesn't
# permute states, we use the IDENTITY as λ̂_8. The unitarisation corrections
# via μ(8,i) coefficients then account for the actual λ_8 entries.
# This is consistent with the paper's approach where non-unitary parts
# are handled by the M gate and B rotations.
l8 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
], dtype=complex)

UNITARY_ADJUSTED_MATRICES: List[np.ndarray] = [l1, l2, l3, l4, l5, l6, l7, l8]
"""List of 8 unitary-adjusted matrices λ̂_1 through λ̂_8."""


# =============================================================================
# Helper Functions
# =============================================================================

def is_unitary(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if a matrix is unitary.
    
    A matrix U is unitary if U†U = UU† = I.
    
    Args:
        matrix: Square numpy array to check.
        tol: Tolerance for numerical comparison.
        
    Returns:
        True if the matrix is unitary within tolerance.
        
    Example:
        >>> is_unitary(UNITARY_ADJUSTED_MATRICES[0])
        True
    """
    identity = np.eye(matrix.shape[0], dtype=complex)
    product = matrix.conj().T @ matrix
    return np.allclose(product, identity, atol=tol)


def expand_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Embed a 3×3 matrix into 4×4 by adding an unused |11⟩ level.
    
    This is required for encoding 3-dimensional qutrit states (colour)
    into 2 qubits (4-dimensional Hilbert space). The |11⟩ state is
    left invariant (acts as identity on this level).
    
    Args:
        mat: A 3×3 complex numpy array.
        
    Returns:
        A 4×4 complex numpy array with mat in the upper-left 3×3 block
        and a 1 in the (3,3) position.
        
    Example:
        >>> expanded = expand_matrix(UNITARY_ADJUSTED_MATRICES[0])
        >>> expanded.shape
        (4, 4)
        >>> expanded[3, 3]
        (1+0j)
    """
    N = mat.shape[0]  # should be 3
    expanded = np.zeros((N + 1, N + 1), dtype=complex)
    expanded[:N, :N] = mat
    expanded[N, N] = 1
    return expanded


def su3_structure_constants() -> Dict[Tuple[int, int, int], complex]:
    """
    Compute the SU(3) structure constants f_abc.
    
    The structure constants are defined by the commutation relations:
        [T_a, T_b] = i f_abc T_c
    where T_a = λ_a / 2 are the generators in fundamental representation.
    
    With standard normalization Tr(T_a T_b) = (1/2) δ_ab.
    
    Returns:
        Dictionary mapping (a, b, c) tuples (1-indexed) to f_abc values.
        Only non-zero entries are included.
        
    Example:
        >>> f = su3_structure_constants()
        >>> f[(1, 2, 3)]  # Should be 1.0
        (1+0j)
    """
    # T_a = λ_a / 2
    Ts = [0.5 * L for L in GELL_MANN_MATRICES]
    
    f: Dict[Tuple[int, int, int], complex] = {}
    
    for a in range(8):
        for b in range(8):
            # Compute [T_a, T_b]
            comm = Ts[a] @ Ts[b] - Ts[b] @ Ts[a]
            
            for c in range(8):
                # f_abc = (2/i) Tr([T_a, T_b] T_c) = -2i Tr([T_a, T_b] T_c)
                val = (-2j) * np.trace(comm @ Ts[c])
                
                if abs(val) > 1e-10:
                    # Use 1-indexed labels (standard physics convention)
                    f[(a + 1, b + 1, c + 1)] = val
                    
    return f


# Module-level cached structure constants
_STRUCTURE_CONSTANTS: Dict[Tuple[int, int, int], complex] | None = None


def get_structure_constants() -> Dict[Tuple[int, int, int], complex]:
    """
    Get cached SU(3) structure constants.
    
    Returns:
        Dictionary of structure constants (computed once and cached).
    """
    global _STRUCTURE_CONSTANTS
    if _STRUCTURE_CONSTANTS is None:
        _STRUCTURE_CONSTANTS = su3_structure_constants()
    return _STRUCTURE_CONSTANTS
