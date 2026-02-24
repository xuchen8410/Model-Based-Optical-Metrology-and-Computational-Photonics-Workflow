"""
Inverse fitting workflow
=======================

This module implements the closed‑loop inverse problem: given a
synthetic reflectance spectrum from a binary grating, recover the
structural parameters (height and duty cycle) by least‑squares
fitting.  Synthetic measurements are generated using the RCWA solver
and Gaussian noise is added.  The SciPy optimisation routines are
used to minimise the sum of squared residuals between measurement and
model.  A simple covariance estimate for the fitted parameters is
provided based on the Jacobian at the optimum.

The workflow can be executed from the command line via ``cpw fit
config.yml``.  A typical configuration file looks like this:

.. code-block:: yaml

    output_dir: results/inverse_fit
    true_height: 100e-9
    true_duty_cycle: 0.5
    wavelength_start: 400e-9
    wavelength_stop: 800e-9
    num_wavelengths: 50
    noise_std: 0.005
    n_ambient: 1.0
    n_substrate: 1.0
    n_grating_high: 3.5
    n_grating_low: 1.5
    polarization: TE
    fourier_order: 3
    initial_guess:
      height: 80e-9
      duty_cycle: 0.4
    bounds:
      height: [50e-9, 150e-9]
      duty_cycle: [0.2, 0.8]

The workflow creates ``output_dir`` if necessary, saves the measured
data and fitted model predictions to CSV files, generates plots
comparing measured and fitted spectra and the residuals, and returns
the fitted parameters along with estimated uncertainties.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Iterable

import numpy as np
import pandas as pd
import yaml

try:
    from scipy.optimize import least_squares
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "SciPy is required for the inverse fitting workflow. Please install scipy."
    ) from e

from ..solvers.rcwa_1d import rcwa_1d
from ..viz import plots


@dataclass
class FitResult:
    height: float
    duty_cycle: float
    residual: float
    covariance: np.ndarray
    standard_errors: Tuple[float, float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "height": self.height,
            "duty_cycle": self.duty_cycle,
            "residual": self.residual,
            "covariance": self.covariance.tolist(),
            "standard_errors": list(self.standard_errors),
        }


def _generate_synthetic(
    wavelengths: np.ndarray,
    true_height: float,
    true_duty_cycle: float,
    period: float,
    n_ambient: float,
    n_substrate: float,
    n_grating_high: float,
    n_grating_low: float,
    polarization: str,
    fourier_order: int,
    noise_std: float,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic measurement and model spectrum.

    Returns a tuple ``(measurement, noiseless)``.  Noise is Gaussian
    with standard deviation ``noise_std`` and fixed random seed for
    reproducibility.
    """
    result = rcwa_1d(
        wavelengths,
        period=period,
        duty_cycle=true_duty_cycle,
        height=true_height,
        n_ambient=n_ambient,
        n_substrate=n_substrate,
        n_grating_high=n_grating_high,
        n_grating_low=n_grating_low,
        polarization=polarization,
        fourier_order=fourier_order,
    )
    R_true = result["R"]
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, size=R_true.shape)
    measurement = R_true + noise
    # Clip measurement into physical bounds
    measurement = np.clip(measurement, 0.0, 1.0)
    return measurement, R_true


def _residuals(
    params: np.ndarray,
    wavelengths: np.ndarray,
    measurement: np.ndarray,
    period: float,
    n_ambient: float,
    n_substrate: float,
    n_grating_high: float,
    n_grating_low: float,
    polarization: str,
    fourier_order: int,
) -> np.ndarray:
    """Compute residual vector between measurement and model for optimisation."""
    height, duty_cycle = params
    model = rcwa_1d(
        wavelengths,
        period=period,
        duty_cycle=duty_cycle,
        height=height,
        n_ambient=n_ambient,
        n_substrate=n_substrate,
        n_grating_high=n_grating_high,
        n_grating_low=n_grating_low,
        polarization=polarization,
        fourier_order=fourier_order,
    )
    return model["R"] - measurement


def run_inverse_fit(config: Dict[str, Any]) -> FitResult:
    """Run the inverse fitting workflow.

    The ``config`` dictionary must contain the keys described in the
    module docstring.  The function returns a ``FitResult`` instance.
    Side effects include writing measurement and fitted data to CSV
    files, generating PNG plots, and printing a summary to stdout.
    """
    output_dir = config.get("output_dir", "fit_output")
    os.makedirs(output_dir, exist_ok=True)

    wl_start = float(config["wavelength_start"])
    wl_stop = float(config["wavelength_stop"])
    num_wl = int(config.get("num_wavelengths", 50))
    wavelengths = np.linspace(wl_start, wl_stop, num_wl)

    period = float(config["period"])
    true_height = float(config["true_height"])
    true_duty = float(config["true_duty_cycle"])
    n_ambient = float(config.get("n_ambient", 1.0))
    n_substrate = float(config.get("n_substrate", 1.0))
    n_high = float(config["n_grating_high"])
    n_low = float(config["n_grating_low"])
    polarization = config.get("polarization", "TE")
    fourier_order = int(config.get("fourier_order", 3))
    noise_std = float(config.get("noise_std", 0.0))
    seed = int(config.get("noise_seed", 0))

    # Generate synthetic measurement
    measurement, true_spectrum = _generate_synthetic(
        wavelengths=wavelengths,
        true_height=true_height,
        true_duty_cycle=true_duty,
        period=period,
        n_ambient=n_ambient,
        n_substrate=n_substrate,
        n_grating_high=n_high,
        n_grating_low=n_low,
        polarization=polarization,
        fourier_order=fourier_order,
        noise_std=noise_std,
        seed=seed,
    )

    # Initial guess
    init_guess_cfg = config.get("initial_guess", {})
    height_guess = float(init_guess_cfg.get("height", true_height))
    duty_guess = float(init_guess_cfg.get("duty_cycle", true_duty))
    x0 = np.array([height_guess, duty_guess], dtype=np.float64)

    # Bounds
    bounds_cfg = config.get("bounds", {})
    height_bounds = bounds_cfg.get("height", [0.0, np.inf])
    duty_bounds = bounds_cfg.get("duty_cycle", [0.0, 1.0])
    lower = [float(height_bounds[0]), float(duty_bounds[0])]
    upper = [float(height_bounds[1]), float(duty_bounds[1])]

    # Optimisation
    result = least_squares(
        _residuals,
        x0,
        args=(
            wavelengths,
            measurement,
            period,
            n_ambient,
            n_substrate,
            n_high,
            n_low,
            polarization,
            fourier_order,
        ),
        bounds=(lower, upper),
        method="trf",
    )

    fitted_height, fitted_duty = result.x
    residual_vector = result.fun
    residual_norm = np.sqrt(np.mean(residual_vector ** 2))

    # Approximate covariance of fitted parameters
    # J: Jacobian matrix; compute (J^T J)^{-1} times variance estimate
    if result.jac is not None and result.jac.size >= 4:
        J = result.jac
        # Estimate variance of residuals
        sigma2 = np.var(residual_vector, ddof=2)
        try:
            cov = np.linalg.inv(J.T @ J) * sigma2
        except np.linalg.LinAlgError:
            cov = np.full((2, 2), np.nan)
    else:
        cov = np.full((2, 2), np.nan)
    if np.any(np.isnan(cov)):
        se_height = se_duty = np.nan
    else:
        se_height = float(np.sqrt(cov[0, 0]))
        se_duty = float(np.sqrt(cov[1, 1]))

    fit_res = FitResult(
        height=float(fitted_height),
        duty_cycle=float(fitted_duty),
        residual=float(residual_norm),
        covariance=cov,
        standard_errors=(se_height, se_duty),
    )

    # Save measurement and fit data
    data_df = pd.DataFrame(
        {
            "wavelength": wavelengths,
            "measurement": measurement,
            "true_spectrum": true_spectrum,
        }
    )
    data_df.to_csv(os.path.join(output_dir, "measurement.csv"), index=False)

    fitted_spectrum = rcwa_1d(
        wavelengths,
        period=period,
        duty_cycle=fitted_duty,
        height=fitted_height,
        n_ambient=n_ambient,
        n_substrate=n_substrate,
        n_grating_high=n_high,
        n_grating_low=n_low,
        polarization=polarization,
        fourier_order=fourier_order,
    )["R"]
    fit_df = pd.DataFrame(
        {
            "wavelength": wavelengths,
            "fitted_spectrum": fitted_spectrum,
        }
    )
    fit_df.to_csv(os.path.join(output_dir, "fit.csv"), index=False)

    # Generate plots
    out_base = os.path.join(output_dir, "inverse_fit")
    residual_plot = measurement - fitted_spectrum
    plots.plot_fit_comparison(
        wavelengths=wavelengths,
        measured=measurement,
        fitted=fitted_spectrum,
        residual=residual_plot,
        out_base=out_base,
        title_prefix="Inverse Fit",
    )

    # Print summary
    summary = (
        f"Inverse fitting completed.\n"
        f"True height: {true_height:.3e} m, True duty cycle: {true_duty:.3f}\n"
        f"Fitted height: {fitted_height:.3e} m ± {se_height:.3e} m\n"
        f"Fitted duty cycle: {fitted_duty:.3f} ± {se_duty:.3f}\n"
        f"Residual (RMSE): {residual_norm:.3e}\n"
    )
    print(summary)
    return fit_res


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file for the inverse fitting workflow."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config