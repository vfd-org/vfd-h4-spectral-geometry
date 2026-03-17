# Vibrational Field Dynamics — Research Series

This repository contains a series of working papers exploring the geometric and algebraic structure of symmetry-constrained field dynamics on H₄ geometry (the 600-cell).

## Papers

### Paper I — Symmetry-Constrained Resonant Modes on H₄ Geometry
**Focus:** Geometric framework and dynamical behaviour

A real scalar field is defined on the vertex graph of the 600-cell — the terminal regular polytope of the non-crystallographic Coxeter family H₄. The Laplacian spectrum contains only 9 distinct eigenvalues across 120 degrees of freedom. Numerical simulations demonstrate persistent localisation, coherent shell reflection, and recurrent breathing dynamics that are absent in degree-matched control graphs.

- [Paper I (LaTeX source)](papers/paper-001-h4-dynamics/vfd_h4_resonance_model.tex)
- [Simulation figures](papers/paper-001-h4-dynamics/figures/)

### Paper II — Algebraic Invariants of the H₄ Laplacian Spectrum
**Focus:** Exact spectral structure and algebraic constraints

The internal organisation of the nine-sector Laplacian spectrum is analysed. Seven structural invariants are identified: φ-cancellation, degree anchoring, conjugate pairing, square multiplicity folding, three-zone partition, spectral backbone decomposition, and departure structure. These invariants are consistent with the coherent dynamics reported in Paper I.

- [Paper II (LaTeX source)](papers/paper-002-h4-invariants/paper2_h4_invariants.tex)
- [Spectral figures](papers/paper-002-h4-invariants/figures/)

## Relationship

```
Paper I  establishes the phenomenon  (coherent dynamics on H₄ geometry)
Paper II establishes the structure   (spectral invariants governing the spectrum)
```

Together they demonstrate that the 600-cell graph possesses a non-generic spectral algebra that produces structured nonlinear dynamics not observed in comparable graphs.

## Reproducibility

All simulations and spectral computations are available in [`/simulations`](simulations/). The code includes:

- 600-cell graph construction and verification
- Laplacian eigenvalue computation
- Nonlinear dynamics simulation (symplectic integrator)
- Control graph generation and comparison
- Spectral and localisation diagnostics

Requirements: Python 3.8+, NumPy, SciPy, Matplotlib, PyTorch (optional, for accelerated integration).

## Key Results

| Result | Paper | Value |
|--------|-------|-------|
| Distinct eigenvalues | I, II | 9 (across 120 modes) |
| Localisation enhancement | I | 5× vs controls |
| Shell cross-correlation | I | 0.68 (vs 0.1 controls) |
| Beat frequencies | I | 36 (vs 7000+ controls) |
| Square multiplicity sequence | II | 1², 2², 3², 4², 5², 6² |
| Spectral backbone | II | 88 = 1 + 3 × 29 |
| φ-cancellation | II | Σb = 0, Σ(b×mult) = 0 |

## Status

These are working papers. No direct physical interpretation is claimed. The results establish a non-generic spectral structure that warrants further mathematical investigation.

## Author

Lee Smart
Institute of Vibrational Field Dynamics
contact@vibrationalfielddynamics.org
[@vfd_org](https://twitter.com/vfd_org)
