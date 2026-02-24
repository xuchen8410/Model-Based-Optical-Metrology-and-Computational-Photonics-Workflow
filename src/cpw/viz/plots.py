"""
Visualization utilities
======================

This module centralizes all plotting functions used in the project.  It
relies solely on matplotlib and does not set explicit colours; the
default matplotlib style is used.  Each plotting function creates a
new figure, draws the requested data, labels the axes, sets a title
where appropriate, and saves the figure to disk.  No subplots are
used because each plot is intended to stand on its own.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Mapping

import matplotlib
matplotlib.use("Agg")  # Use a non‑interactive backend for script execution
import matplotlib.pyplot as plt


def plot_reflectance(
    wavelengths: Iterable[float],
    reflectance: Iterable[float],
    *,
    title: str = "Reflectance Spectrum",
    out_path: str,
) -> None:
    """Plot a reflectance spectrum and save it to a PNG file.

    Parameters
    ----------
    wavelengths : iterable of float
        Wavelengths on the x‑axis (metres).
    reflectance : iterable of float
        Corresponding reflectance values (0 to 1).
    title : str, optional
        Title for the figure.  Defaults to "Reflectance Spectrum".
    out_path : str
        Path to save the PNG file.  Parent directories are created
        automatically.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(wavelengths, reflectance)
    ax.set_xlabel("Wavelength (m)")
    ax.set_ylabel("Reflectance")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(
    wavelengths: Iterable[float],
    order_to_reflectance: Mapping[int, Iterable[float]],
    *,
    out_path: str,
    title: str = "Fourier Order Convergence",
) -> None:
    """Plot convergence of reflectance with Fourier order.

    Each order's reflectance is plotted on the same axes.  A legend
    indicates the order.  The figure is saved to the specified
    ``out_path``.

    Parameters
    ----------
    wavelengths : iterable of float
        Wavelengths on the x‑axis (metres).
    order_to_reflectance : mapping of int to iterable of float
        Keys are Fourier orders; values are reflectance arrays for each
        wavelength.
    out_path : str
        Path to save the PNG file.
    title : str, optional
        Title for the figure.  Defaults to "Fourier Order Convergence".
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots()
    for order, refl in sorted(order_to_reflectance.items()):
        ax.plot(wavelengths, refl, label=f"order {order}")
    ax.set_xlabel("Wavelength (m)")
    ax.set_ylabel("Reflectance")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fit_comparison(
    wavelengths: Iterable[float],
    measured: Iterable[float],
    fitted: Iterable[float],
    residual: Iterable[float],
    *,
    out_base: str,
    title_prefix: str = "Inverse Fit",
) -> None:
    """Plot measured vs fitted spectra and residual curves.

    Two separate figures are generated.  The first overlays the
    measured and fitted reflectance spectra; the second plots the
    residual (measured minus fitted) as a function of wavelength.
    Filenames are constructed by appending ``_fit.png`` and
    ``_residual.png`` to ``out_base``.

    Parameters
    ----------
    wavelengths : iterable of float
        Wavelengths on the x‑axis (metres).
    measured : iterable of float
        Measured reflectance values.
    fitted : iterable of float
        Fitted reflectance values.
    residual : iterable of float
        Difference between measured and fitted (same length as
        ``wavelengths``).
    out_base : str
        Base path (without extension) for saving the plots.
    title_prefix : str, optional
        Prefix for the figure titles.  ``"Inverse Fit"`` by default.
    """
    # Plot measured vs fitted spectrum
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig1, ax1 = plt.subplots()
    ax1.plot(wavelengths, measured, label="measured")
    ax1.plot(wavelengths, fitted, label="fitted")
    ax1.set_xlabel("Wavelength (m)")
    ax1.set_ylabel("Reflectance")
    ax1.set_title(f"{title_prefix}: measured vs fitted")
    ax1.set_ylim(0.0, 1.05)
    ax1.legend()
    fig1.savefig(out_base + "_fit.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # Plot residual curve
    fig2, ax2 = plt.subplots()
    ax2.plot(wavelengths, residual)
    ax2.set_xlabel("Wavelength (m)")
    ax2.set_ylabel("Residual (measured - fitted)")
    ax2.set_title(f"{title_prefix}: residuals")
    ax2.axhline(0.0, linestyle="--", linewidth=0.5)
    fig2.savefig(out_base + "_residual.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)