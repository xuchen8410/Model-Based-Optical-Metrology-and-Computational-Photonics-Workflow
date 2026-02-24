"""
Tests for the inverse fitting workflow.

The test generates a synthetic measurement with known parameters and
verifies that the fitting routine recovers them within a reasonable
relative tolerance.  The tolerance is deliberately loose because the
noise and simplified forward model limit the achievable accuracy.
"""

import os
import sys
import numpy as np

# Ensure the package can be imported without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from cpw.workflows.inverse_fit import run_inverse_fit


def test_inverse_fit_recovery(tmp_path):
    # Define true parameters and generate synthetic data via the workflow
    config = {
        "output_dir": str(tmp_path / "fit_test"),
        "true_height": 100e-9,
        "true_duty_cycle": 0.5,
        "period": 600e-9,
        "n_ambient": 1.0,
        "n_substrate": 1.0,
        "n_grating_high": 3.5,
        "n_grating_low": 1.5,
        "wavelength_start": 400e-9,
        "wavelength_stop": 800e-9,
        "num_wavelengths": 40,
        "noise_std": 0.005,
        "noise_seed": 123,
        "polarization": "TE",
        "fourier_order": 3,
        "initial_guess": {"height": 80e-9, "duty_cycle": 0.4},
        "bounds": {"height": [50e-9, 150e-9], "duty_cycle": [0.2, 0.8]},
    }
    result = run_inverse_fit(config)
    # Check that fitted parameters are within 10% relative error
    assert abs(result.height - config["true_height"]) / config["true_height"] < 0.1
    assert abs(result.duty_cycle - config["true_duty_cycle"]) / config["true_duty_cycle"] < 0.1