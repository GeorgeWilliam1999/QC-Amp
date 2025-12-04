"""
Tests for circuit builders.

Tests cover:
- R_quark_prep circuit properties
- R_gluon_prep circuit properties
- quark_emission_absorption circuit structure
"""

import numpy as np
import pytest
from qiskit import QuantumRegister
from qiskit.quantum_info import Operator, Statevector

from qc_amp.circuits import (
    R_quark_prep,
    R_gluon_prep,
    quark_emission_absorption,
    create_registers,
)


TOL = 1e-10


class TestRQuarkPrep:
    """Tests for quark singlet preparation circuit."""
    
    def test_unitary(self):
        """R_q circuit should be unitary."""
        q = QuantumRegister(2, "q")
        qtil = QuantumRegister(2, "qtil")
        circ = R_quark_prep(q, qtil)
        op = Operator(circ)
        assert op.is_unitary(), "R_q is not unitary"
    
    def test_singlet_state(self):
        """R_q should prepare entangled colour singlet-like state."""
        q = QuantumRegister(2, "q")
        qtil = QuantumRegister(2, "qtil")
        circ = R_quark_prep(q, qtil)
        
        psi_in = Statevector.from_label("0000")
        psi_out = psi_in.evolve(circ)
        out = psi_out.to_dict()
        
        # Expect (1/√3)(|0000⟩ + |0101⟩ + |1010⟩)
        expected_keys = {"0000", "0101", "1010"}
        amp = 1 / np.sqrt(3)
        
        assert set(out.keys()) == expected_keys, \
            "R_q output support on wrong basis states"
        
        for k in expected_keys:
            assert np.isclose(out[k], amp, atol=1e-8), \
                f"R_q amplitude for {k} is wrong"


class TestRGluonPrep:
    """Tests for gluon superposition preparation circuit."""
    
    def test_unitary(self):
        """R_g circuit should be unitary."""
        g = QuantumRegister(3, "g")
        circ = R_gluon_prep(g)
        op = Operator(circ)
        assert op.is_unitary(), "R_g is not unitary"
    
    def test_equal_superposition(self):
        """R_g should prepare equal superposition over 8 states."""
        g = QuantumRegister(3, "g")
        circ = R_gluon_prep(g)
        
        psi_in = Statevector.from_label("000")
        psi_out = psi_in.evolve(circ)
        probs = psi_out.probabilities()
        
        # Expect uniform over 8 states
        expected = np.ones(8) / 8
        assert np.allclose(probs, expected, atol=1e-8), \
            "R_g does not give uniform superposition"


class TestQuarkEmissionAbsorption:
    """Tests for the full quark-gluon diagram circuit."""
    
    def test_creates_circuit(self):
        """Should create a valid QuantumCircuit."""
        circ = quark_emission_absorption()
        assert circ is not None
        assert circ.num_qubits > 0
    
    def test_correct_num_qubits(self):
        """Circuit should have correct number of qubits."""
        circ = quark_emission_absorption(n_vertices=2)
        # 3 (gluon) + 2 (unitarisation for n_vertices=2) + 2 (quark) + 2 (antiquark) = 9
        assert circ.num_qubits == 9
    
    def test_variable_vertices(self):
        """Should handle different numbers of vertices."""
        for n_v in [1, 2, 3, 4]:
            circ = quark_emission_absorption(n_vertices=n_v)
            assert circ is not None


class TestCreateRegisters:
    """Tests for register creation utility."""
    
    def test_returns_dict(self):
        """Should return a dictionary of registers."""
        regs = create_registers()
        assert isinstance(regs, dict)
    
    def test_required_keys(self):
        """Should include all required register keys."""
        regs = create_registers()
        required = {'gluon', 'quark', 'anti_quark', 'unitarisation'}
        assert required.issubset(set(regs.keys()))
    
    def test_register_sizes(self):
        """Registers should have correct sizes."""
        regs = create_registers()
        assert len(regs['gluon']) == 3
        assert len(regs['quark']) == 2
        assert len(regs['anti_quark']) == 2
