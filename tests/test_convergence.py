"""
Tests for Fourier order convergence in the RCWA solver.

The convergence test ensures that as the Fourier truncation order
increases the reflectance approaches its asymptotic value in a
monotonic fashion.  The test uses a reference order (highest
computed) as the benchmark and verifies that the error decreases
monotonically with the order.
"""

import os
import sys
import numpy as np

# Ensure the package can be imported without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from cpw.solvers.rcwa_1d import convergence_scan


def test_convergence_monotonicity():
    # Define a grating and wavelength array
    wavelengths = np.linspace(450e-9, 650e-9, 8)
    period = 550e-9
    duty = 0.4
    height = 100e-9
    n_ambient = 1.0
    n_substrate = 1.0
    n_high = 3.0
    n_low = 1.5
    polarization = "TE"

    max_order = 6
    conv = convergence_scan(
        wavelengths,
        period=period,
        duty_cycle=duty,
        height=height,
        n_ambient=n_ambient,
        n_substrate=n_substrate,
        n_grating_high=n_high,
        n_grating_low=n_low,
        polarization=polarization,
        max_order=max_order,
    )
    # Use the highest order as reference
    ref = conv[max_order]
    # For each lower order, compute the maximum absolute difference
    diffs = []
    for order in range(max_order):
        err = np.max(np.abs(conv[order] - ref))
        diffs.append(err)
    # Ensure that the errors decrease monotonically
    for i in range(len(diffs) - 1):
        assert diffs[i + 1] <= diffs[i] + 1e-12, "Convergence is not monotonic"