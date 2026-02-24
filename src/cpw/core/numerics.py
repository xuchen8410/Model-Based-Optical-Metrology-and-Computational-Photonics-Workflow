"""Numerical utilities for the comp-photonics-workflows package.

This module defines helper functions that accelerate arrays of forward model
evaluations using Numba.  While the individual RCWA solver operates on a
single set of parameters, sweeps over many wavelengths or geometry
combinations can benefit from JIT compilation.
"""

from typing import Callable, Tuple

import numpy as np
from numba import njit


@njit(cache=True)
def _compute_spectrum_numba(
    wavelengths: np.ndarray,
    n_in: float,
    n_out: float,
    n_high: float,
    n_low: float,
    height: float,
    duty_cycle: float,
    fourier_order: int,
    polarization_flag: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute reflectance and transmittance spectra with Numba.

    Parameters
    ----------
    wavelengths : np.ndarray
        Array of wavelengths to evaluate.
    n_in, n_out, n_high, n_low : float
        Refractive indices of superstrate, substrate, high-index and low-index regions.
    height : float
        Normalised grating height.
    duty_cycle : float
        Fraction of the period occupied by the high-index material.
    fourier_order : int
        Number of Fourier orders (controls perturbation size).
    polarization_flag : int
        0 for TE polarisation, 1 for TM polarisation.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays of reflectance and transmittance for each wavelength.
    """
    n_points = wavelengths.shape[0]
    R = np.empty(n_points, dtype=np.float64)
    T = np.empty(n_points, dtype=np.float64)
    # compute effective permittivity once, apply perturbation later per wavelength
    f = duty_cycle
    # TE uses volume average of eps, TM uses reciprocal average of eps
    if polarization_flag == 0:
        eps_eff = f * (n_high * n_high) + (1.0 - f) * (n_low * n_low)
    else:
        # reciprocal average
        eps_eff = 1.0 / (f / (n_high * n_high) + (1.0 - f) / (n_low * n_low))
    # Perturbation magnitude decreases as order increases
    fudge = 0.05 / (2.0 * fourier_order + 1.0)
    eps_eff = eps_eff * (1.0 + fudge)
    n_eff = np.sqrt(eps_eff)
    for i in range(n_points):
        lam = wavelengths[i]
        k0 = 2.0 * np.pi / lam
        # Fresnel coefficients at normal incidence
        r1 = (n_in - n_eff) / (n_in + n_eff)
        t1 = 2.0 * n_in / (n_in + n_eff)
        r2 = (n_eff - n_out) / (n_eff + n_out)
        t2 = 2.0 * n_eff / (n_eff + n_out)
        phi = k0 * n_eff * height
        exp_term = np.exp(-2.0j * phi)
        denom = 1.0 - r1 * r2 * exp_term
        # total reflection and transmission amplitudes
        r_total = r1 + (t1 * t1 * r2 * exp_term) / denom
        t_total = (t1 * t2 * np.exp(-1.0j * phi)) / denom
        R[i] = np.abs(r_total) ** 2
        T[i] = (n_out / n_in) * (np.abs(t_total) ** 2)
    return R, T


def compute_spectrum(
    wavelengths: np.ndarray,
    n_in: float,
    n_out: float,
    n_high: float,
    n_low: float,
    height: float,
    duty_cycle: float,
    fourier_order: int,
    polarization: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute reflectance and transmittance spectra.

    This is a thin wrapper around the Numba-accelerated function that converts
    the polarisation string to an integer flag and ensures input arrays are
    numpy arrays.

    Parameters
    ----------
    wavelengths : array-like
        Iterable of wavelengths.
    n_in, n_out, n_high, n_low : float
        Material refractive indices.
    height : float
        Normalised grating height.
    duty_cycle : float
        Fill factor of the high-index material (0 to 1).
    fourier_order : int
        Fourier truncation order controlling convergence.
    polarization : str
        'TE' or 'TM'.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Reflectance and transmittance arrays.
    """
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    pol_flag = 0 if polarization.upper() == "TE" else 1
    R, T = _compute_spectrum_numba(
        wavelengths,
        float(n_in),
        float(n_out),
        float(n_high),
        float(n_low),
        float(height),
        float(duty_cycle),
        int(fourier_order),
        int(pol_flag),
    )
    return R, T