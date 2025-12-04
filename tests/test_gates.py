"""
Tests for quantum gate implementations.

Tests cover:
- R gate properties and action
- A (increment) gate functionality
- B1 and B gate unitarity
- μ coefficient values
- Lambda gate action
- M and Q gate unitarity
- G' and G gate structure
"""

import numpy as np
import pytest
from qiskit import QuantumRegister
from qiskit.quantum_info import Operator, Statevector

from qc_amp.gates import (
    R_MATRIX,
    R_GATE,
    A_gate,
    B1_gate,
    B_gate,
    mu_coefficient,
    quark_colour_bits,
    Lambda_gate,
    M_gate,
    Q_gate,
    G_prime_gate,
    G_gate,
)
from qc_amp.su3 import (
    GELL_MANN_MATRICES,
    UNITARY_ADJUSTED_MATRICES,
    expand_matrix,
)


TOL = 1e-10


def _bitstr(k: int, n: int) -> str:
    """Convert integer k to n-bit binary string."""
    return format(k, f"0{n}b")


class TestRGate:
    """Tests for the R gate (quark singlet preparation)."""
    
    def test_unitary(self):
        """R matrix should be unitary."""
        op = Operator(R_MATRIX)
        assert op.is_unitary(), "R is not unitary"
    
    def test_shape(self):
        """R matrix should be 4×4."""
        assert R_MATRIX.shape == (4, 4)
    
    def test_action_on_00(self):
        """R|00⟩ should give equal superposition of colour states."""
        e00 = np.array([1, 0, 0, 0], dtype=complex)
        out00 = R_MATRIX @ e00
        
        # Should be (1/√3)(|00⟩ + |01⟩ + |10⟩)
        amp_expected = 1 / np.sqrt(3)
        assert np.allclose(out00[:3], amp_expected, atol=1e-8)
        assert abs(out00[3]) < TOL, "R|00⟩ leaks into |11⟩"
    
    def test_fixes_11(self):
        """R should leave |11⟩ invariant."""
        e11 = np.array([0, 0, 0, 1], dtype=complex)
        out11 = R_MATRIX @ e11
        assert np.allclose(out11, e11, atol=TOL)


class TestAGate:
    """Tests for the A (increment) gate."""
    
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_unitary(self, n):
        """A gate should be unitary for different register sizes."""
        U = QuantumRegister(n, "U")
        Acirc = A_gate(U)
        Aop = Operator(Acirc)
        assert Aop.is_unitary(), f"A is not unitary for n={n}"
    
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_increment_action(self, n):
        """A should map |k⟩ → |k+1 mod 2^n⟩."""
        U = QuantumRegister(n, "U")
        Acirc = A_gate(U)
        
        for k in range(2**n):
            in_state = Statevector.from_label(_bitstr(k, n))
            out_state = in_state.evolve(Acirc).to_dict()
            
            k_next = (k + 1) % (2**n)
            key_next = _bitstr(k_next, n)
            
            assert len(out_state) == 1 and key_next in out_state, \
                f"A does not map |{k}⟩ → |{k_next}⟩ for n={n}"


class TestB1Gate:
    """Tests for the B1(α) single-qubit gate."""
    
    @pytest.mark.parametrize("alpha", [0.0, 0.3, 0.5, 0.6+0.2j])
    def test_unitary(self, alpha):
        """B1(α) should be unitary for valid α."""
        gate = B1_gate(alpha)
        op = Operator(gate)
        assert op.is_unitary(), f"B1({alpha}) is not unitary"
    
    def test_invalid_alpha_raises(self):
        """B1(α) should raise ValueError if |α|² > 1."""
        with pytest.raises(ValueError):
            B1_gate(1.5)
    
    def test_determinant(self):
        """B1(α) should have determinant 1 (special unitary)."""
        alpha = 0.4
        gate = B1_gate(alpha)
        mat = Operator(gate).data
        det = np.linalg.det(mat)
        assert np.isclose(abs(det), 1, atol=TOL)


class TestBGate:
    """Tests for the B(α) gate on unitarisation register."""
    
    @pytest.mark.parametrize("NU", [1, 2, 3])
    def test_unitary(self, NU):
        """B(α) should be unitary for different register sizes."""
        U = QuantumRegister(NU, "U")
        alpha = 0.4
        Bcirc = B_gate(alpha, U)
        op = Operator(Bcirc)
        assert op.is_unitary(), f"B({alpha}) is not unitary for NU={NU}"


class TestMuCoefficient:
    """Tests for the μ(a,i) coefficient function."""
    
    def test_standard_cases(self):
        """Test μ(a,i) for a < 8 based on row equality."""
        for a in range(1, 8):
            for i in range(1, 4):
                l_unit = UNITARY_ADJUSTED_MATRICES[a - 1]
                l_orig = GELL_MANN_MATRICES[a - 1]
                row_equal = np.allclose(l_unit[i - 1, :] - l_orig[i - 1, :], 0)
                mu_val = mu_coefficient(a, i)
                
                if row_equal:
                    assert np.isclose(mu_val, 0.5, atol=TOL), \
                        f"μ({a},{i}) should be 1/2"
                else:
                    assert abs(mu_val) < TOL, f"μ({a},{i}) should be 0"
    
    def test_lambda8_special_cases(self):
        """Test special μ values for λ_8."""
        assert np.isclose(mu_coefficient(8, 1), 1 / (2 * np.sqrt(3)), atol=TOL)
        assert np.isclose(mu_coefficient(8, 2), 1 / (2 * np.sqrt(3)), atol=TOL)
        assert np.isclose(mu_coefficient(8, 3), -1 / np.sqrt(3), atol=TOL)


class TestQuarkColourBits:
    """Tests for quark colour encoding."""
    
    def test_colour_mapping(self):
        """Test colour index to bit string mapping."""
        assert quark_colour_bits(1) == "00"
        assert quark_colour_bits(2) == "01"
        assert quark_colour_bits(3) == "10"
    
    def test_invalid_colour_raises(self):
        """Invalid colour index should raise ValueError."""
        with pytest.raises(ValueError):
            quark_colour_bits(4)
        with pytest.raises(ValueError):
            quark_colour_bits(0)


class TestLambdaGate:
    """Tests for the Λ (Lambda) gate."""
    
    def test_unitary(self):
        """Λ gate should be unitary."""
        gluon = QuantumRegister(3, "g")
        quark = QuantumRegister(2, "q")
        circ = Lambda_gate(quark, gluon)
        op = Operator(circ)
        assert op.is_unitary(), "Lambda is not unitary"
    
    def test_action_on_colour_states(self):
        """Test Λ applies correct transformation for each gluon colour."""
        gluon = QuantumRegister(3, "g")
        quark = QuantumRegister(2, "q")
        circ = Lambda_gate(quark, gluon)
        
        n_g = len(gluon)
        n_q = len(quark)
        
        for a, lhat in enumerate(UNITARY_ADJUSTED_MATRICES, start=1):
            lam_expanded = expand_matrix(lhat)
            
            for b in range(4):
                # Build |q=b, g=a-1⟩
                q_bits = _bitstr(b, n_q)
                g_bits = _bitstr(a - 1, n_g)
                label_in = q_bits + g_bits
                psi_in = Statevector.from_label(label_in)
                
                # Evolve with Lambda
                psi_out = psi_in.evolve(circ)
                out_dict = psi_out.to_dict()
                
                # Expected quark state: lam_expanded |b⟩
                e_b = np.zeros(4, dtype=complex)
                e_b[b] = 1
                q_expected = lam_expanded @ e_b
                
                # Verify amplitudes
                for key, amp in out_dict.items():
                    qk = key[:n_q]
                    gk = key[n_q:]
                    
                    if gk != g_bits:
                        assert abs(amp) < 1e-8, \
                            f"Lambda leaked gluon state for a={a}, b={b}"
                    else:
                        j = int(qk, 2)
                        assert np.isclose(amp, q_expected[j], atol=1e-8), \
                            f"Lambda wrong on (a={a}, b={b}), basis {key}"


class TestMGate:
    """Tests for the M gate."""
    
    def test_unitary(self):
        """M gate should be unitary."""
        gluon = QuantumRegister(3, "g")
        quark = QuantumRegister(2, "q")
        U = QuantumRegister(2, "U")
        circ = M_gate(gluon, quark, U)
        op = Operator(circ)
        assert op.is_unitary(), "M is not unitary"


class TestQGate:
    """Tests for the Q gate."""
    
    def test_unitary(self):
        """Q gate should be unitary."""
        gluon = QuantumRegister(3, "g")
        quark = QuantumRegister(2, "q")
        U = QuantumRegister(2, "U")
        Qc = Q_gate(gluon, quark, U)
        op = Operator(Qc)
        assert op.is_unitary(), "Q_gate is not unitary"


class TestGGates:
    """Tests for triple-gluon gates G' and G."""
    
    def test_G_prime_unitary(self):
        """G' gate should be unitary."""
        g1 = QuantumRegister(3, "g1")
        g2 = QuantumRegister(3, "g2")
        g3 = QuantumRegister(3, "g3")
        U = QuantumRegister(2, "U")
        
        circ = G_prime_gate(g1, g2, g3, U)
        op = Operator(circ)
        assert op.is_unitary(), "G' is not unitary"
    
    def test_G_unitary(self):
        """G gate should be unitary."""
        g1 = QuantumRegister(3, "g1")
        g2 = QuantumRegister(3, "g2")
        g3 = QuantumRegister(3, "g3")
        U = QuantumRegister(2, "U")
        
        circ = G_gate(g1, g2, g3, U)
        op = Operator(circ)
        assert op.is_unitary(), "G is not unitary"
