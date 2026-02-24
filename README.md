# Model-Based-Optical-Metrology-and-Computational-Photonics-Workflow
A reproducible computational photonics toolkit that connects:  Maxwell’s equations → numerical model → validated implementation → structural parameter inference  This project implements a validated 1D RCWA forward solver with automated convergence analysis, parallel parameter sweeps, and measurement-driven inverse fitting.
comp-photonics-workflows/
│
├── pyproject.toml
├── README.md
├── LICENSE
│
├── src/cpw/
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── materials.py
│   │   ├── grids.py
│   │   ├── numerics.py
│   │
│   ├── solvers/
│   │   ├── rcwa_1d.py
│   │   ├── fdtd_2d_te.py (future extension)
│   │   ├── bpm_2d.py (future extension)
│   │
│   ├── workflows/
│   │   ├── sweep.py
│   │   ├── inverse_fit.py
│   │
│   ├── io/
│   │   ├── config.py
│   │   ├── results.py
│   │
│   ├── viz/
│   │   ├── plots.py
│   │
│   └── cli.py
│
├── examples/
│   ├── rcwa_grating_sweep.yml
│   ├── inverse_fit_example.yml
│
├── tests/
│   ├── test_energy_conservation.py
│   ├── test_convergence.py
│   ├── test_inverse_fit.py
│
├── docs/
│   ├── theory_rcwa.md
│   ├── numerics_stability.md
│   ├── measurement_link.md
│
└── .github/workflows/
    └── ci.yml
