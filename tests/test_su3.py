"""
Tests for SU(3) group theory utilities.

Tests cover:
- Gell-Mann matrices properties (Hermitian, traceless)
- Unitary-adjusted matrices (unitarity)
- Matrix expansion function
- Structure constants computation
"""

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from qc_amp.su3 import (
    GELL_MANN_MATRICES,
    UNITARY_ADJUSTED_MATRICES,
    expand_matrix,
    is_unitary,
    su3_structure_constants,
    get_structure_constants,
)


TOL = 1e-10


class TestGellMannMatrices:
    """Tests for Gell-Mann matrices properties."""
    
    def test_count(self):
        """Should have exactly 8 Gell-Mann matrices."""
        assert len(GELL_MANN_MATRICES) == 8
    
    @pytest.mark.parametrize("a", range(8))
    def test_hermitian(self, a):
        """Each Gell-Mann matrix should be Hermitian (L = L†)."""
        L = GELL_MANN_MATRICES[a]
        assert np.allclose(L, L.conj().T, atol=TOL), f"L{a+1} is not Hermitian"
    
    @pytest.mark.parametrize("a", range(8))
    def test_traceless(self, a):
        """Each Gell-Mann matrix should be traceless."""
        L = GELL_MANN_MATRICES[a]
        assert abs(np.trace(L)) < TOL, f"L{a+1} is not traceless"
    
    @pytest.mark.parametrize("a", range(8))
    def test_shape(self, a):
        """Each Gell-Mann matrix should be 3×3."""
        L = GELL_MANN_MATRICES[a]
        assert L.shape == (3, 3), f"L{a+1} has wrong shape"


class TestUnitaryAdjustedMatrices:
    """Tests for unitary-adjusted matrices."""
    
    def test_count(self):
        """Should have exactly 8 unitary-adjusted matrices."""
        assert len(UNITARY_ADJUSTED_MATRICES) == 8
    
    @pytest.mark.parametrize("a", range(8))
    def test_unitary(self, a):
        """Each unitary-adjusted matrix should be unitary."""
        l = UNITARY_ADJUSTED_MATRICES[a]
        I3 = np.eye(3, dtype=complex)
        assert np.allclose(l.conj().T @ l, I3, atol=TOL), f"l{a+1} is not unitary"
    
    @pytest.mark.parametrize("a", range(8))
    def test_shape(self, a):
        """Each unitary-adjusted matrix should be 3×3."""
        l = UNITARY_ADJUSTED_MATRICES[a]
        assert l.shape == (3, 3), f"l{a+1} has wrong shape"


class TestExpandMatrix:
    """Tests for the matrix expansion function."""
    
    @pytest.mark.parametrize("a", range(8))
    def test_preserves_unitarity(self, a):
        """Expanded matrices should remain unitary."""
        l = UNITARY_ADJUSTED_MATRICES[a]
        expanded = expand_matrix(l)
        op = Operator(expanded)
        assert op.is_unitary(), f"expanded l{a+1} is not unitary"
    
    @pytest.mark.parametrize("a", range(8))
    def test_correct_shape(self, a):
        """Expanded matrices should be 4×4."""
        l = UNITARY_ADJUSTED_MATRICES[a]
        expanded = expand_matrix(l)
        assert expanded.shape == (4, 4), f"expanded l{a+1} has wrong shape"
    
    @pytest.mark.parametrize("a", range(8))
    def test_fixes_state_11(self, a):
        """Expanded matrices should leave |11⟩ invariant."""
        l = UNITARY_ADJUSTED_MATRICES[a]
        expanded = expand_matrix(l)
        e11 = np.array([0, 0, 0, 1], dtype=complex)
        out = expanded @ e11
        assert np.allclose(out, e11, atol=TOL), f"expanded l{a+1} does not fix |11⟩"
    
    @pytest.mark.parametrize("a", range(8))
    def test_upper_block_preserved(self, a):
        """Upper 3×3 block should match original matrix."""
        l = UNITARY_ADJUSTED_MATRICES[a]
        expanded = expand_matrix(l)
        assert np.allclose(expanded[:3, :3], l, atol=TOL)


class TestIsUnitary:
    """Tests for the is_unitary helper function."""
    
    def test_identity_is_unitary(self):
        """Identity matrix should be unitary."""
        I = np.eye(3, dtype=complex)
        assert is_unitary(I)
    
    def test_pauli_x_is_unitary(self):
        """Pauli X matrix should be unitary."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        assert is_unitary(X)
    
    def test_non_unitary_detected(self):
        """Non-unitary matrices should be detected."""
        non_unitary = np.array([[1, 1], [0, 1]], dtype=complex)
        assert not is_unitary(non_unitary)


class TestStructureConstants:
    """Tests for SU(3) structure constants."""
    
    def test_returns_dict(self):
        """Structure constants should be returned as a dictionary."""
        f = su3_structure_constants()
        assert isinstance(f, dict)
    
    def test_antisymmetric(self):
        """f_abc should be totally antisymmetric."""
        f = su3_structure_constants()
        
        for (a, b, c), val in f.items():
            # Check f_abc = -f_bac
            if (b, a, c) in f:
                assert np.isclose(f[(b, a, c)], -val, atol=TOL)
            
            # Check f_abc = -f_acb
            if (a, c, b) in f:
                assert np.isclose(f[(a, c, b)], -val, atol=TOL)
    
    def test_known_values(self):
        """Check some known structure constant values."""
        f = su3_structure_constants()
        
        # f_123 = 1
        assert np.isclose(f.get((1, 2, 3), 0), 1.0, atol=TOL)
        
        # f_147 = 1/2
        assert np.isclose(f.get((1, 4, 7), 0), 0.5, atol=TOL)
    
    def test_caching(self):
        """Cached structure constants should match fresh computation."""
        fresh = su3_structure_constants()
        cached = get_structure_constants()
        
        assert set(fresh.keys()) == set(cached.keys())
        for key in fresh:
            assert np.isclose(fresh[key], cached[key], atol=TOL)
