"""
Command‑line interface for comp‑photonics‑workflows
===================================================

This module defines the ``cpw`` console script entry point.  The CLI
exposes three subcommands:

* ``sweep``: perform a parameter sweep given a YAML configuration file.
* ``converge``: perform a Fourier order convergence scan on a single
  geometry specified in a YAML file.
* ``fit``: generate a synthetic measurement and perform an inverse
  fitting of the grating parameters.

Usage examples:

.. code-block:: shell

    python -m cpw.sweep examples/sweep_example.yml
    cpw sweep examples/sweep_example.yml
    cpw converge examples/sweep_example.yml
    cpw fit examples/inverse_fit_example.yml

Each subcommand prints a short summary to the console and writes
results and figures to an output directory specified in the config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .workflows import sweep as sweep_module
from .workflows import inverse_fit as inverse_fit_module
from .solvers import rcwa_1d as rcwa_solver
from .viz import plots


def _cmd_sweep(args: argparse.Namespace) -> None:
    config = sweep_module.load_config(args.config)
    df = sweep_module.run_sweep(config)
    # Print summary
    print(f"Sweep completed. {len(df)} rows written to {config.get('output_dir', 'sweep_output')}")


def _cmd_converge(args: argparse.Namespace) -> None:
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Use first height and duty cycle in config
    heights = cfg.get("heights", [cfg.get("height", None)])
    duty_cycles = cfg.get("duty_cycles", [cfg.get("duty_cycle", None)])
    if not heights or not duty_cycles:
        raise ValueError("Convergence requires at least one height and duty_cycle in config")
    height = float(heights[0])
    duty = float(duty_cycles[0])
    wl_start = float(cfg["wavelength_start"])
    wl_stop = float(cfg["wavelength_stop"])
    num_wl = int(cfg.get("num_wavelengths", 50))
    wavelengths = [wl_start + i * (wl_stop - wl_start) / (num_wl - 1) for i in range(num_wl)]
    period = float(cfg["period"])
    n_ambient = float(cfg.get("n_ambient", 1.0))
    n_substrate = float(cfg.get("n_substrate", 1.0))
    n_high = float(cfg["n_grating_high"])
    n_low = float(cfg["n_grating_low"])
    polarization = cfg.get("polarization", "TE")
    max_order = int(cfg.get("max_order", cfg.get("fourier_order", 5)))
    out_dir = cfg.get("output_dir", "convergence_output")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    conv = rcwa_solver.convergence_scan(
        wavelengths=wavelengths,
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
    # Save convergence data
    import pandas as pd
    rows = []
    for order, refl in conv.items():
        for wl, r_val in zip(wavelengths, refl):
            rows.append({"order": order, "wavelength": wl, "reflectance": r_val})
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "convergence.csv", index=False)
    # Plot convergence
    conv_plot = Path(out_dir) / "convergence.png"
    plots.plot_convergence(
        wavelengths=wavelengths,
        order_to_reflectance=conv,
        out_path=str(conv_plot),
        title=f"Convergence (h={height:.1e} m, duty={duty:.2f})",
    )
    print(f"Convergence scan completed. Results saved in {out_dir}")


def _cmd_fit(args: argparse.Namespace) -> None:
    config = inverse_fit_module.load_config(args.config)
    result = inverse_fit_module.run_inverse_fit(config)
    # Print fitted parameters (already printed in workflow)
    _ = result  # just for clarity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="comp-photonics-workflows CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # sweep subcommand
    parser_sweep = subparsers.add_parser("sweep", help="run a parameter sweep")
    parser_sweep.add_argument("config", type=str, help="YAML configuration file for sweep")
    parser_sweep.set_defaults(func=_cmd_sweep)

    # converge subcommand
    parser_conv = subparsers.add_parser("converge", help="run a convergence scan")
    parser_conv.add_argument("config", type=str, help="YAML configuration file for convergence")
    parser_conv.set_defaults(func=_cmd_converge)

    # fit subcommand
    parser_fit = subparsers.add_parser("fit", help="run inverse fitting")
    parser_fit.add_argument("config", type=str, help="YAML configuration file for inverse fitting")
    parser_fit.set_defaults(func=_cmd_fit)

    args = parser.parse_args(argv)
    # Dispatch to subcommand function
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())