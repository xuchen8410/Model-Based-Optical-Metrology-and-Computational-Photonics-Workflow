# comp-photonics-workflows

A self-contained example of a **closed-loop computational photonics workflow**.

This repository demonstrates how modern **model-based optical metrology** systems recover nanoscale structural parameters from far-field scattering measurements.

It implements:

- Minimal 1D RCWA-like forward model  
- Synthetic spectrum generation  
- Nonlinear inverse fitting  
- Fourier-order convergence analysis  
- Energy conservation validation  
- Parallel parameter sweeps  

The code is compact, readable, and physics-driven.

---

# Industrial Motivation | 工业背景

In semiconductor manufacturing, **critical dimension (CD)** and **overlay** must be controlled at the nanometre scale.

Direct imaging methods such as SEM or AFM are:

- Slow  
- Expensive  
- Potentially destructive  

Modern alternative → **Optical Scatterometry (Diffraction-Based Overlay, DBO)**

Core principle:

1. Measure far-field diffraction spectrum  
2. Simulate spectrum using physics-based model  
3. Solve inverse scattering problem  
4. Recover structural parameters  

Precision comes from **accurate forward modeling**, not from direct spatial resolution.

---

# Numerical Model | 数值模型

We simulate a **1D binary grating** under normal incidence.

Parameters:

- Period: Lambda = 1 (normalized)  
- Refractive indices: n_high, n_low  
- Duty cycle: f  
- Height: h  
- Polarization: TE and TM  

---

## Minimal RCWA-like Forward Model

A full RCWA solver normally includes:

- Fourier expansion of epsilon(x)  
- Convolution matrix construction  
- Eigenmode decomposition  
- Boundary matching  

To emphasize clarity, we implement a simplified physical model.

### 1) Zeroth-order effective permittivity

For TE polarization:

epsilon_eff = f * n_high^2 + (1 - f) * n_low^2  

For TM polarization:

epsilon_eff = 1 / (f / n_high^2 + (1 - f) / n_low^2)

This captures dominant harmonic behavior.

---

### 2) Fourier truncation mimic

We introduce a perturbation:

delta_epsilon proportional to 1 / (2N + 1)

As Fourier order N increases:

- Perturbation decreases  
- Reflectance converges monotonically  

This mimics real RCWA convergence behavior.

---

### 3) Effective slab approximation

The grating is treated as a homogeneous slab:

n_eff = sqrt(epsilon_eff)

Reflection and transmission are computed using:

- Fresnel equations  
- Thin-film interference  

Energy conservation:

R + T approximately equals 1 (lossless case)

---

# Convergence & Validation

Real RCWA solvers show:

- Different convergence rates for TE and TM  
- Slow convergence for metallic gratings  

This repository includes:

- `convergence_scan()`  
- `energy_check()`  
- Pytest validation for:
  - Energy conservation  
  - Monotonic convergence  
  - Inverse recovery consistency  

Ensuring physically consistent behavior.

---

# Synthetic Measurement & Inverse Fitting

Scatterometry is an inverse problem:

Forward:
geometry → spectrum  

Inverse:
spectrum → geometry  

Implemented functions:

### generate_synthetic_measurement()

- Forward simulation  
- Add Gaussian noise  
- Deterministic random seed  

### fit_structure()

- SciPy `least_squares` (Levenberg–Marquardt)  
- Recover:
  - Grating height  
  - Duty cycle  
- Jacobian-based covariance estimate  
- Confidence intervals  

Closed-loop workflow:

simulate → add noise → fit → recover → quantify uncertainty

---

# Parallelization

Workflows support multiprocessing via:

`concurrent.futures.ProcessPoolExecutor`

Features:

- YAML-configurable worker count  
- Deterministic seeds per worker  
- Suitable for:
  - Parameter sweeps  
  - Batch inverse solves  

---

