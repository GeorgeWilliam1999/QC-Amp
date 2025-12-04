"""
Tests for colour factor computation.

Tests cover:
- Basic colour factor extraction
- Known analytical results
- Verification utilities
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit

from qc_amp.circuits import quark_emission_absorption
from qc_amp.colour_factors import (
    compute_colour_factor,
    compute_colour_factor_detailed,
    verify_colour_factor,
    format_colour_factor_result,
)


class TestComputeColourFactor:
    """Tests for colour factor computation."""
    
    def test_returns_complex(self):
        """Should return a complex number."""
        circ = quark_emission_absorption()
        C = compute_colour_factor(circ)
        assert isinstance(C, (complex, np.complexfloating))
    
    def test_quark_emission_absorption_value(self):
        """Test colour factor for single gluon emission/absorption."""
        circ = quark_emission_absorption(n_vertices=2)
        C = compute_colour_factor(circ, n_quarks=1, n_gluons=1)
        
        # Expected: C = 4 (Casimir C_F = 4/3, times 3 for colour average)
        # Actually for this diagram structure the expected value is 4
        assert np.isclose(C.real, 4.0, atol=1e-6), \
            f"Expected colour factor ~4, got {C}"


class TestComputeColourFactorDetailed:
    """Tests for detailed colour factor computation."""
    
    def test_returns_tuple(self):
        """Should return (C, amplitude, N) tuple."""
        circ = quark_emission_absorption()
        result = compute_colour_factor_detailed(circ)
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    def test_normalization_factor(self):
        """Normalization factor should match formula."""
        circ = quark_emission_absorption()
        _, _, N = compute_colour_factor_detailed(circ, n_quarks=1, n_gluons=1, N_c=3)
        
        # N = N_c^{n_q} × (N_c² - 1)^{n_g} = 3 × 8 = 24
        expected_N = 3 * 8
        assert N == expected_N
    
    def test_consistency(self):
        """Detailed and simple compute should give same C."""
        circ = quark_emission_absorption()
        C_simple = compute_colour_factor(circ)
        C_detailed, _, _ = compute_colour_factor_detailed(circ)
        assert np.isclose(C_simple, C_detailed)


class TestVerifyColourFactor:
    """Tests for verification utility."""
    
    def test_correct_value_passes(self):
        """Correct value should pass verification."""
        assert verify_colour_factor(4.0 + 0j, expected=4.0)
    
    def test_wrong_value_fails(self):
        """Wrong value should fail verification."""
        assert not verify_colour_factor(3.0 + 0j, expected=4.0, tolerance=0.1)
    
    def test_with_tolerance(self):
        """Should respect tolerance parameter."""
        assert verify_colour_factor(4.1 + 0j, expected=4.0, tolerance=0.05)
        assert not verify_colour_factor(4.1 + 0j, expected=4.0, tolerance=0.01)


class TestFormatColourFactorResult:
    """Tests for result formatting."""
    
    def test_returns_string(self):
        """Should return a formatted string."""
        circ = quark_emission_absorption()
        result = format_colour_factor_result(circ)
        assert isinstance(result, str)
    
    def test_contains_key_info(self):
        """Output should contain key information."""
        circ = quark_emission_absorption()
        result = format_colour_factor_result(circ)
        
        assert "Normalization" in result or "N =" in result
        assert "amplitude" in result.lower() or "Amplitude" in result
        assert "Colour factor" in result or "colour factor" in result
    
    def test_with_expected_value(self):
        """Should include comparison when expected value provided."""
        circ = quark_emission_absorption()
        result = format_colour_factor_result(circ, expected=4.0)
        
        assert "Expected" in result
        assert "error" in result.lower()
