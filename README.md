# Model-Based-Optical-Metrology-and-Computational-Photonics-Workflow
This is an RCWA-based metrology workflow that links spectral measurements to structural parameters.
The system includes convergence validation, automated parameter sweeps, and inverse fitting with uncertainty estimation.
This is directly applicable to scatterometry-driven process control in semiconductor manufacturing.

This reproducible computational photonics toolkit that connects:  Maxwell’s equations → numerical model → validated implementation → structural parameter inference  This project implements a validated 1D RCWA forward solver with automated convergence analysis, parallel parameter sweeps, and measurement-driven inverse fitting.

Architecture: RCWA-based forward solver + automated inversion engine：结构参数 → RCWA → 光谱
测量光谱 → 反演拟合 → CD / height / duty cycle
系统理解并实现了基于 RCWA 的周期结构电磁仿真与反演工作流，涵盖 Maxwell 方程推导、傅里叶展开与模态耦合矩阵构建、S-matrix 数值稳定传播、Fourier 阶数收敛验证与能量守恒检测。在高入射角场景下分析了倏逝波增强、Rayleigh anomaly 与 Gibbs 现象对数值稳定性的影响，并通过阶数扫描与矩阵条件数监测提升结果可信度。构建了可并行化批量参数扫描与光谱反演流程，用于结构参数（CD/刻蚀深度）估计


comp-photonics-workflows/
├── README.md
├── pyproject.toml
├── examples/
│   ├── sweep_example.yml
│   └── inverse_fit_example.yml
├── src/
│   └── cpw/
│       ├── __init__.py
│       ├── cli.py
│       ├── core/
│       │   └── numerics.py
│       ├── solvers/
│       │   └── rcwa_1d.py
│       ├── workflows/
│       │   ├── sweep.py
│       │   └── inverse_fit.py
│       └── viz/
│           └── plots.py
├── tests/
│   ├── test_energy_conservation.py
│   ├── test_convergence.py
│   └── test_inverse_fit.py
└── docs/         # 占位目录，可根据需要添加文档
