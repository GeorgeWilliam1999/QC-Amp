# QC-Amp: Quantum Computing for QCD Amplitude Calculations

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Qiskit-based library for computing **colour factors** in QCD scattering amplitudes using quantum circuits.

## Overview

QC-Amp implements quantum algorithms for calculating colour factors that appear in gauge theory scattering amplitudes. The package provides:

- **SU(3) group theory utilities**: Gell-Mann matrices, structure constants, unitary adjustments
- **Quantum gate implementations**: Quark-gluon vertices (Q), triple-gluon vertices (G), unitarisation machinery
- **Circuit builders**: State preparation and full Feynman diagram circuits
- **Colour factor extraction**: Compute colour factors from circuit amplitudes

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/GeorgeWilliam/QC-Amp.git
cd QC-Amp
pip install -e ".[dev,notebooks]"
```

### Dependencies only

```bash
pip install qiskit numpy
```

## Quick Start

```python
from qc_amp import compute_colour_factor
from qc_amp.circuits import quark_emission_absorption

# Build circuit for quark emitting and absorbing a gluon
circuit = quark_emission_absorption(n_vertices=2)

# Compute the colour factor
C = compute_colour_factor(circuit, n_quarks=1, n_gluons=1)
print(f"Colour factor: {C.real:.4f}")  # Expected: 4.0
```

## Package Structure

```
QC-Amp/
├── src/qc_amp/
│   ├── __init__.py         # Public API
│   ├── su3.py              # SU(3) group theory
│   ├── gates.py            # Quantum gate definitions
│   ├── circuits.py         # Circuit builders
│   └── colour_factors.py   # Colour factor computation
├── tests/
│   ├── test_su3.py
│   ├── test_gates.py
│   ├── test_circuits.py
│   └── test_colour_factors.py
├── examples/
│   └── colour_factors_demo.ipynb
├── pyproject.toml
└── README.md
```

## API Reference

### SU(3) Utilities (`qc_amp.su3`)

```python
from qc_amp.su3 import (
    GELL_MANN_MATRICES,       # List of 8 Gell-Mann matrices λ_a
    UNITARY_ADJUSTED_MATRICES, # List of 8 unitary-adjusted matrices λ̂_a
    su3_structure_constants,   # Compute f_abc
    expand_matrix,             # Embed 3×3 → 4×4
)
```

### Quantum Gates (`qc_amp.gates`)

| Gate | Function | Description |
|------|----------|-------------|
| R | `R_GATE` | Quark singlet preparation |
| A | `A_gate(U)` | Increment on unitarisation register |
| B | `B_gate(α, U)` | Controlled rotation |
| Λ | `Lambda_gate(q, g)` | Gluon-controlled colour rotation |
| M | `M_gate(g, q, U)` | Unitarisation correction |
| Q | `Q_gate(g, q, U)` | Complete quark-gluon vertex |
| G' | `G_prime_gate(g1, g2, g3, U)` | Structure constant rotations |
| G | `G_gate(g1, g2, g3, U)` | Complete triple-gluon vertex |

### Circuit Builders (`qc_amp.circuits`)

```python
from qc_amp.circuits import (
    R_quark_prep,           # Prepare quark-antiquark singlet
    R_gluon_prep,           # Prepare gluon superposition
    quark_emission_absorption,  # Full diagram circuit
)
```

### Colour Factor Computation (`qc_amp.colour_factors`)

```python
from qc_amp.colour_factors import (
    compute_colour_factor,         # Simple interface
    compute_colour_factor_detailed, # Returns (C, amplitude, N)
    verify_colour_factor,          # Check against expected value
)
```

## Examples

### Visualize Circuit Components

```python
from qiskit import QuantumRegister
from qc_amp.gates import Lambda_gate, Q_gate

# Create registers
gluon = QuantumRegister(3, 'g')
quark = QuantumRegister(2, 'q')
U = QuantumRegister(3, 'U')

# Draw Lambda gate
lambda_circ = Lambda_gate(quark, gluon)
lambda_circ.draw('mpl')

# Draw Q gate
q_circ = Q_gate(gluon, quark, U)
q_circ.draw('mpl')
```

### Compute Colour Factor with Details

```python
from qc_amp.circuits import quark_emission_absorption
from qc_amp.colour_factors import format_colour_factor_result

circuit = quark_emission_absorption()
print(format_colour_factor_result(circuit, expected=4.0))
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=qc_amp

# Run specific test file
pytest tests/test_gates.py -v
```

## Theory Background

The colour factor $C$ for a Feynman diagram is computed as:

$$C = N \cdot \langle\Omega_{all}|\psi_{final}\rangle$$

where:
- $N = N_c^{n_q} \cdot (N_c^2 - 1)^{n_g}$ is the normalization
- $n_q$ = number of quark lines
- $n_g$ = number of gluon lines
- $N_c = 3$ for SU(3) (QCD)
- $|\Omega_{all}\rangle = |0\cdots0\rangle$ is the reference state

The quantum circuit implements the diagram structure with:
1. State preparation gates ($R_g$, $R_q$)
2. Interaction vertices ($Q$ for quark-gluon, $G$ for triple-gluon)
3. Inverse preparation gates

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## References

This package implements the quantum algorithms described in:

> **H. A. Chawdhry and M. Pellen**, "Quantum simulation of colour in perturbative quantum chromodynamics",
> *SciPost Phys.* **15**, 205 (2023). [arXiv:2303.04818](https://arxiv.org/abs/2303.04818)

```bibtex
@article{Chawdhry:2023yxy,
    author = "Chawdhry, Hitesh A. and Pellen, Mathieu",
    title = "{Quantum simulation of colour in perturbative quantum chromodynamics}",
    journal = "SciPost Phys.",
    volume = "15",
    pages = "205",
    year = "2023",
    doi = "10.21468/SciPostPhys.15.5.205",
    eprint = "2303.04818",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph"
}
```

### Additional Resources

- [Qiskit Documentation](https://qiskit.org/documentation/)
