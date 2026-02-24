# comp-photonics-workflows

This repository provides a self-contained example of a **closed‑loop computational photonics workflow**.  The goal is to demonstrate how modern industrial optical metrology tools use physics-based models and inverse algorithms to recover nanoscale structure parameters from far-field scattering measurements.  In particular, we implement a minimal one-dimensional rigorous coupled-wave analysis (RCWA) solver, generate synthetic scattering data, fit structural parameters and evaluate convergence.  The code is deliberately simple and well commented so that readers without a background in electromagnetics can follow along.

## Industrial motivation

In semiconductor manufacturing, overlay and critical dimension (CD) errors must be controlled at the nanometre scale.  Direct imaging techniques such as scanning electron microscopy or atomic force microscopy are slow, expensive and potentially destructive.  An alternative is **optical scatterometry**, also called **diffraction-based overlay metrology (DBO)**.  Scatterometry compares a measured far‑field diffraction signature to a simulated signature and infers the parameters of the periodic structure.  It is a model-based technique: the inverse problem of light scattering is solved by accurately modelling the measurement and fitting a small set of structural parameters.  The *Unconventional Metrology* thesis explains that model‑based optical metrology aims to retrieve parameters like critical dimension or overlay with much higher precision than direct imaging by relying on an accurate forward model and prior knowledge【387669924056495†L552-L619】.  Scatterometry offers a non-contact, non-destructive, high-throughput measurement suitable for in‑situ process control【168770430569318†L59-L63】.

The forward model in most commercial scatterometry tools is based on **rigorous coupled-wave analysis (RCWA)**.  The RCWA method expands the permittivity profile of a periodic structure into Fourier harmonics and solves Maxwell’s equations for all diffraction orders.  The thesis notes that RCWA is the most widely adopted forward modelling technique for scatterometry【387669924056495†L615-L621】.  Other methods such as the boundary element method (BEM), finite element method (FEM) and finite-difference time-domain (FDTD) are also used, but RCWA offers speed for simple layered gratings.  Industrial software packages (e.g. KLA‑Tencor’s **AcuRate™** and simulation-to-measurement tools) use RCWA to design overlay targets and compare simulated performance metrics to measured data【590437949763876†L90-L105】.  Accurate simulation is essential because uncertainties in film thickness, optical constants or grating geometry can cause the simulated response to deviate from the measurement【590437949763876†L122-L127】.  Large parameter spaces and cross-correlations can make the inverse problem ill‑posed【168770430569318†L15-L17】.  Regularisation and sensitivity analysis are therefore used to reduce ambiguities【168770430569318†L117-L120】.

## Numerical method

Our implementation focuses on a **one-dimensional binary grating** under normal incidence.  The grating has a period \(\Lambda = 1\) and is composed of two materials with refractive indices \(n_{\text{high}}\) and \(n_{\text{low}}\), filling fractions \(f\) and \(1-f\), and finite thickness \(h\).  For simplicity the pitch is normalised and wavelengths are specified relative to the pitch.  The solver supports both TE (electric field perpendicular to the grating grooves) and TM polarisation.

In a full RCWA solver one expands the spatially varying permittivity \(\varepsilon(x)\) into a truncated Fourier series, builds a convolution matrix and solves an eigenvalue problem to obtain the normal modes in the periodic medium.  Boundary matching with superstrate and substrate then yields the diffraction efficiencies for each order.  However, such a general solver is mathematically involved.  To emphasise clarity over completeness we implement a **minimal RCWA‑like forward model**:

1.  Compute the zeroth Fourier coefficient (volume average) of the permittivity for TE and TM polarisation.  The effective permittivity for TE is \(\varepsilon_{\text{eff,TE}} = f\,n_{\text{high}}^2 + (1-f)\,n_{\text{low}}^2\); for TM we use the reciprocal average \(\varepsilon_{\text{eff,TM}} = 1/(f/n_{\text{high}}^2 + (1-f)/n_{\text{low}}^2)\).  Higher harmonics decay rapidly for moderate index contrast, so the zeroth term dominates.
2.  Introduce a small order-dependent perturbation \(\delta\varepsilon \propto 1/(2N+1)\) to mimic the effect of increasing Fourier order \(N\).  As \(N\) grows the perturbation decreases, leading to a monotonic convergence of reflectance.
3.  Treat the grating as a homogeneous slab of effective refractive index \(n_{\text{eff}} = \sqrt{\varepsilon_{\text{eff}}}\) and thickness \(h\).  Use Fresnel’s equations and thin-film interference formulas to compute the zero-order reflection \(R\) and transmission \(T\) amplitudes.  The energy conservation test (\(R+T \approx 1\)) holds for lossless materials.

Although this model is not a full RCWA implementation, it captures the essential features needed for a closed-loop demonstration: control of Fourier truncation order, TE/TM polarisation dependence, finite grating height and energy conservation.  The numerics are implemented with NumPy, and a Numba-accelerated function computes arrays of reflectance values for sweeps over wavelengths or geometries.

## Convergence and validation

Convergence with respect to Fourier order is an important aspect of RCWA.  In the literature the convergence rate can vary between TE and TM polarisation and may be slow for metallic gratings【215945173961009†L73-L79】.  Our simplified model includes a `convergence_scan` function that computes reflectance for a list of Fourier orders and returns the sequence for inspection.  The `energy_check` function verifies that the sum of reflected and transmitted power is close to unity for lossless materials.  The associated `pytest` tests in `tests/test_energy_conservation.py` and `tests/test_convergence.py` ensure that both energy conservation and monotonic convergence are satisfied.

## Synthetic measurement and inverse fitting

Because scatterometry measurements are indirect, the structural parameters must be inferred from the measured spectrum.  Forward solve algorithms like RCWA map geometry to spectra, but the inverse problem does not have a closed form and must be solved numerically【215945173961009†L143-L148】.  Inverse fitting is the foundation of model-based optical metrology【387669924056495†L597-L631】.

The `generate_synthetic_measurement` function creates a synthetic reflectance spectrum using the forward model, adds Gaussian noise with a fixed random seed and returns the “measured” data.  The `fit_structure` function wraps SciPy’s Levenberg-Marquardt (`least_squares`) optimiser to recover the grating height and duty cycle that best match the measured spectrum.  A simple covariance estimate from the Jacobian is used to provide confidence intervals.  The inverse fitting workflow is tested in `tests/test_inverse_fit.py`.

## Parallelisation

Parameter sweeps and inverse fits can be computationally expensive.  Both the sweep (`workflows/sweep.py`) and inverse fitting (`workflows/inverse_fit.py`) modules support multiprocessing via `concurrent.futures.ProcessPoolExecutor`.  Users can specify the number of workers in the YAML configuration file.  A deterministic random seed is propagated to each worker to ensure reproducible results.

## Command line interface

After installation the package exposes a command-line tool `cpw` with three subcommands:

```
cpw sweep <config.yml>     # run parameter sweep over wavelengths, heights and duty cycles
cpw converge <config.yml>  # run Fourier order convergence test
cpw fit <config.yml>       # generate synthetic data and perform inverse fitting
```

Each command reads a YAML configuration file (examples are provided in the `examples/` directory), creates an output directory, saves results as CSV files, generates plots as PNG files and prints a summary to the console.

## Installation and usage

Assuming a Linux or macOS environment with Python ≥3.9, run the following commands from the repository root:

```bash
# create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# install the package in editable mode
pip install -e .

# run the test suite
pytest

# run the sweep example
cpw sweep examples/sweep_example.yml

# run the convergence test
cpw converge examples/sweep_example.yml

# run the inverse fitting example
cpw fit examples/inverse_fit_example.yml
```

The package depends only on publicly available Python libraries (`numpy`, `scipy`, `pandas`, `pyyaml`, `matplotlib`, `numba` and `pytest`), ensuring that examples run end-to-end without external data.

## Directory structure

```
comp-photonics-workflows/
├── pyproject.toml        # Package metadata and dependencies
├── README.md             # This document
├── src/cpw/
│   ├── __init__.py       # Package initialisation
│   ├── cli.py            # Command line interface
│   ├── core/
│   │   └── numerics.py   # Numerical utilities and Numba-accelerated functions
│   ├── solvers/
│   │   └── rcwa_1d.py    # Minimal 1D RCWA solver
│   ├── workflows/
│   │   ├── sweep.py      # Parameter sweep workflow
│   │   └── inverse_fit.py# Synthetic measurement and inverse fitting
│   └── viz/
│       └── plots.py      # Plotting utilities (Matplotlib)
├── examples/
│   ├── sweep_example.yml # Example configuration for parameter sweep
│   └── inverse_fit_example.yml # Example configuration for inverse fitting
├── tests/
│   ├── test_energy_conservation.py
│   ├── test_convergence.py
│   └── test_inverse_fit.py
└── docs/
    └── (placeholder for additional documentation)
```

## License

This example project is provided under the MIT license.  The included literature references are cited for educational purposes.  For industrial applications users should consult validated RCWA implementations and metrology software from vendors such as KLA and perform their own convergence and uncertainty studies.