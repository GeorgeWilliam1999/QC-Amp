"""
Circuit Builders for QCD Amplitude Calculations

This module provides high-level circuit construction functions,
as described in Chawdhry & Pellen, SciPost Phys. 15, 205 (2023).

State Preparation:
    - R_quark_prep: Prepare quark-antiquark colour singlet state (Eq. 5, Appendix A)
    - R_gluon_prep: Prepare gluon in equal superposition of colours (Eq. 4)

Feynman Diagram Circuits:
    - quark_emission_absorption: Two Q gates on single quark line (Fig. 1)

These circuits implement the quantum algorithms for computing
colour factors in QCD scattering amplitudes.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from qc_amp.gates import R_GATE, Q_gate, R_MATRIX

__all__ = [
    "R_quark_prep",
    "R_gluon_prep",
    "quark_emission_absorption",
]


def R_quark_prep(
    quark: QuantumRegister,
    anti_quark: QuantumRegister
) -> QuantumCircuit:
    """
    Prepare quark-antiquark pair in colour singlet-like state.
    
    Maps:
        |00⟩|00⟩ → (1/√3)(|00⟩|00⟩ + |01⟩|01⟩ + |10⟩|10⟩)
    
    This creates an entangled state where the quark and antiquark
    always have the same colour, representing the colour structure
    of a quark propagator.
    
    Args:
        quark: QuantumRegister for quark colour (2 qubits).
        anti_quark: QuantumRegister for antiquark colour (2 qubits).
        
    Returns:
        QuantumCircuit implementing R_q.
        
    Example:
        >>> q = QuantumRegister(2, 'q')
        >>> qbar = QuantumRegister(2, 'qbar')
        >>> prep = R_quark_prep(q, qbar)
    """
    qc = QuantumCircuit(quark, anti_quark, name="R_q")
    
    # Apply R gate to quark register
    qc.append(R_GATE, quark)
    
    # Entangle antiquark with quark via CNOTs
    qc.cx(quark[0], anti_quark[0])
    qc.cx(quark[1], anti_quark[1])
    
    return qc


def R_gluon_prep(gluon: QuantumRegister) -> QuantumCircuit:
    """
    Prepare gluon in equal superposition over 8 colours.
    
    Maps:
        |000⟩ → (1/√8) ∑_{a=0}^{7} |a⟩
    
    This represents a gluon propagator where all colour indices
    are summed over with equal weight.
    
    Args:
        gluon: QuantumRegister for gluon colour (3 qubits).
        
    Returns:
        QuantumCircuit implementing R_g.
        
    Example:
        >>> g = QuantumRegister(3, 'g')
        >>> prep = R_gluon_prep(g)
    """
    qc = QuantumCircuit(gluon, name="R_g")
    qc.h(gluon)
    return qc


def quark_emission_absorption(n_vertices: int = 2) -> QuantumCircuit:
    """
    Build circuit for quark emitting and absorbing a gluon.
    
    This implements the diagram structure:
        R_g → R_q → Q → Q → R_g† → R_q†
    
    representing a quark line with two gluon vertices (emission
    followed by absorption of the same gluon).
    
    The circuit computes the colour factor for this diagram when
    measured in the |0...0⟩ state.
    
    Args:
        n_vertices: Number of Q gates (vertices) to apply.
                   Default is 2 for emission + absorption.
        
    Returns:
        QuantumCircuit implementing the full diagram.
        
    Example:
        >>> circ = quark_emission_absorption()
        >>> circ.draw()
    """
    # Calculate unitarisation register size
    NU = int(np.ceil(np.log2(n_vertices) + 1))
    
    # Create registers
    g = QuantumRegister(3, 'g')       # gluon
    U = QuantumRegister(NU, 'U')      # unitarisation
    q = QuantumRegister(2, 'q')       # quark
    qtil = QuantumRegister(2, 'qbar') # anti-quark
    
    qc = QuantumCircuit(g, U, q, qtil, name="QuarkGluonDiagram")
    
    # Build gate objects
    Rg_circ = R_gluon_prep(g)
    Rg_gate = Rg_circ.to_gate(label="R_g")
    
    Rq_circ = R_quark_prep(q, qtil)
    Rq_gate = Rq_circ.to_gate(label="R_q")
    
    Q_circ = Q_gate(g, q, U)
    Q_gate_obj = Q_circ.to_gate(label="Q")
    
    # Inverse gates
    Rg_dag = Rg_gate.inverse()
    Rg_dag.label = "R_g†"
    
    Rq_dag = Rq_gate.inverse()
    Rq_dag.label = "R_q†"
    
    # Compose circuit structure
    # 1. Gluon preparation
    qc.append(Rg_gate, g)
    
    # 2. Quark preparation
    qc.append(Rq_gate, list(q) + list(qtil))
    
    # 3. Q gates (vertices)
    for _ in range(n_vertices):
        qc.append(Q_gate_obj, list(g) + list(q) + list(U))
    
    # 4. Inverse gluon preparation
    qc.append(Rg_dag, g)
    
    # 5. Inverse quark preparation  
    qc.append(Rq_dag, list(q) + list(qtil))
    
    return qc


def create_registers(
    n_gluons: int = 1,
    n_quarks: int = 1,
    n_vertices: int = 2
) -> dict:
    """
    Create quantum registers for a colour factor calculation.
    
    Args:
        n_gluons: Number of gluon lines.
        n_quarks: Number of quark lines.
        n_vertices: Number of interaction vertices.
        
    Returns:
        Dictionary with register names as keys and QuantumRegisters as values.
        
    Example:
        >>> regs = create_registers(n_gluons=1, n_quarks=1, n_vertices=2)
        >>> regs['gluon']
        QuantumRegister(3, 'gluon')
    """
    NU = int(np.ceil(np.log2(n_vertices) + 1))
    
    registers = {
        'gluon': QuantumRegister(3, 'gluon'),
        'quark': QuantumRegister(2, 'quark'),
        'anti_quark': QuantumRegister(2, 'anti_quark'),
        'unitarisation': QuantumRegister(NU, 'U'),
    }
    
    return registers
