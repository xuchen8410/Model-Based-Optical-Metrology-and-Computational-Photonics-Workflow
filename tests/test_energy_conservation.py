"""
Tests for energy conservation in the RCWA solver.

These tests verify that the sum of reflected and transmitted power is
unity to within a small numerical tolerance for lossless materials.
"""

import os
import sys
import numpy as np

# Ensure the package can be imported without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from cpw.solvers.rcwa_1d import rcwa_1d, energy_check


def test_energy_conservation_single():
    # Choose some arbitrary parameters for the grating
    wavelengths = np.linspace(400e-9, 800e-9, 10)
    result = rcwa_1d(
        wavelengths,
        period=600e-9,
        duty_cycle=0.5,
        height=150e-9,
        n_ambient=1.0,
        n_substrate=1.0,
        n_grating_high=3.5,
        n_grating_low=1.5,
        polarization="TE",
        fourier_order=4,
    )
    R = result["R"]
    T = result["T"]
    # Assert that R + T == 1 within tolerance
    assert np.allclose(R + T, 1.0, atol=1e-12), "Energy is not conserved"


def test_energy_check_function():
    wavelengths = np.linspace(500e-9, 700e-9, 5)
    err = energy_check(
        wavelengths,
        period=700e-9,
        duty_cycle=0.4,
        height=100e-9,
        n_ambient=1.0,
        n_substrate=1.0,
        n_grating_high=2.0,
        n_grating_low=1.5,
        polarization="TM",
        fourier_order=2,
    )
    assert err < 1e-12, f"Energy conservation error too large: {err}"