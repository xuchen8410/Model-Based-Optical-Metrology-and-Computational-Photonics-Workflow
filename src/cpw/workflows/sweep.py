"""
Parameter sweep workflow
=======================

This module provides a workflow to sweep over geometric parameters of
a binary grating and compute reflectance spectra using the RCWA
solver.  The sweep can explore different heights and duty cycles and
optionally run computations in parallel using multiple processes.
The results are assembled into a pandas DataFrame and written to
disk.  Additionally, for each parameter combination a reflectance
spectrum plot is generated and saved as a PNG file.

The sweep is configured via a YAML file.  A typical configuration
looks like this:

.. code-block:: yaml

    output_dir: results/sweep_run
    period: 600e-9
    heights: [50e-9, 100e-9, 150e-9]
    duty_cycles: [0.3, 0.5, 0.7]
    height_unit: m  # optional metadata
    duty_unit: fraction  # optional metadata
    wavelength_start: 400e-9
    wavelength_stop: 800e-9
    num_wavelengths: 50
    n_ambient: 1.0
    n_substrate: 1.0
    n_grating_high: 3.5
    n_grating_low: 1.5
    polarization: TE
    fourier_order: 3
    num_workers: 2

Users can run the sweep from the command line via ``cpw sweep
config.yml``.  The workflow creates ``output_dir`` if it does not
exist, saves a CSV file ``results.csv`` with all computed data, and
generates one PNG plot per parameter combination.
"""

from __future__ import annotations

import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from typing import Dict, Any, Iterable, Tuple, List, Optional

import numpy as np
import pandas as pd
import yaml

from ..solvers.rcwa_1d import rcwa_1d
from ..viz import plots


def _compute_single_sweep(
    height: float,
    duty: float,
    wavelengths: np.ndarray,
    period: float,
    n_ambient: float,
    n_substrate: float,
    n_grating_high: float,
    n_grating_low: float,
    polarization: str,
    fourier_order: int,
) -> Dict[str, Any]:
    """Helper function to compute a single parameter combination.

    Parameters are as in ``run_sweep``.  Returns a dictionary
    containing the parameters and computed reflectance spectrum.
    """
    result = rcwa_1d(
        wavelengths,
        period=period,
        duty_cycle=duty,
        height=height,
        n_ambient=n_ambient,
        n_substrate=n_substrate,
        n_grating_high=n_grating_high,
        n_grating_low=n_grating_low,
        polarization=polarization,
        fourier_order=fourier_order,
    )
    return {
        "height": height,
        "duty_cycle": duty,
        "R": result["R"],
        "T": result["T"],
    }


def run_sweep(config: Dict[str, Any]) -> pd.DataFrame:
    """Run a parameter sweep using the RCWA solver.

    The ``config`` dictionary must contain the keys documented in the
    module docstring.  The wavelengths array is constructed from
    ``wavelength_start``, ``wavelength_stop`` and ``num_wavelengths``.

    Returns a pandas DataFrame with one row per (height, duty_cycle,
    wavelength) combination and columns ``height``, ``duty_cycle``,
    ``wavelength``, ``reflectance``, ``transmittance``.
    """
    output_dir = config.get("output_dir", "sweep_output")
    os.makedirs(output_dir, exist_ok=True)

    # Construct wavelength array
    wl_start = float(config["wavelength_start"])
    wl_stop = float(config["wavelength_stop"])
    num_wl = int(config.get("num_wavelengths", 50))
    wavelengths = np.linspace(wl_start, wl_stop, num_wl)

    heights = [float(h) for h in config.get("heights", [])]
    duty_cycles = [float(d) for d in config.get("duty_cycles", [])]
    if not heights or not duty_cycles:
        raise ValueError("heights and duty_cycles must be non‑empty in the config")

    period = float(config["period"])
    n_ambient = float(config.get("n_ambient", 1.0))
    n_substrate = float(config.get("n_substrate", 1.0))
    n_grating_high = float(config["n_grating_high"])
    n_grating_low = float(config["n_grating_low"])
    polarization = config.get("polarization", "TE")
    fourier_order = int(config.get("fourier_order", 3))
    num_workers = int(config.get("num_workers", 1))

    # Prepare tasks
    tasks: List[Tuple[float, float]] = list(product(heights, duty_cycles))
    results: List[Dict[str, Any]] = []

    # Define partial for worker
    worker_func = partial(
        _compute_single_sweep,
        wavelengths=wavelengths,
        period=period,
        n_ambient=n_ambient,
        n_substrate=n_substrate,
        n_grating_high=n_grating_high,
        n_grating_low=n_grating_low,
        polarization=polarization,
        fourier_order=fourier_order,
    )

    if num_workers > 1:
        # Use a process pool for parallel execution
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_func, h, d) for h, d in tasks]
            for fut in futures:
                results.append(fut.result())
    else:
        for h, d in tasks:
            results.append(worker_func(h, d))

    # Build DataFrame
    rows = []
    for res in results:
        h = res["height"]
        d = res["duty_cycle"]
        R = res["R"]
        T = res["T"]
        for wl, r_val, t_val in zip(wavelengths, R, T):
            rows.append(
                {
                    "height": h,
                    "duty_cycle": d,
                    "wavelength": wl,
                    "reflectance": r_val,
                    "transmittance": t_val,
                }
            )
        # Generate plot for this combination
        fname = f"reflectance_h{h:.3e}_d{d:.2f}.png"
        out_path = os.path.join(output_dir, fname)
        plots.plot_reflectance(
            wavelengths,
            R,
            title=f"Reflectance (h={h:.1e} m, duty={d:.2f})",
            out_path=out_path,
        )

    df = pd.DataFrame(rows)
    # Save results to CSV
    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    return df


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file for the sweep workflow."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config