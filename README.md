![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status: Research](https://img.shields.io/badge/status-research-blue)

# Vibrational Field Dynamics — Research Series

This repository contains a series of working papers exploring the geometric and algebraic structure of symmetry-constrained field dynamics on H₄ geometry (the 600-cell).

## Papers

### Paper I — Symmetry-Constrained Resonant Modes on H₄ Geometry
**Focus:** Geometric framework and dynamical behaviour

A real scalar field is defined on the vertex graph of the 600-cell — the terminal regular polytope of the non-crystallographic Coxeter family H₄. The Laplacian spectrum contains only 9 distinct eigenvalues across 120 degrees of freedom. Numerical simulations demonstrate persistent localisation, coherent shell reflection, and recurrent breathing dynamics that are absent in degree-matched control graphs.

- [Paper I (PDF)](papers/paper-001-h4-dynamics/Symmetry_Constrained_Resonant_Modes_on_H4_Geometry.pdf)
- [Paper I (LaTeX source)](papers/paper-001-h4-dynamics/vfd_h4_resonance_model.tex)
- [Simulation figures](papers/paper-001-h4-dynamics/figures/)

### Paper II — Algebraic Invariants of the H₄ Laplacian Spectrum
**Focus:** Exact spectral structure and algebraic constraints

The internal organisation of the nine-sector Laplacian spectrum is analysed. Seven structural invariants are identified: φ-cancellation, degree anchoring, conjugate pairing, square multiplicity folding, three-zone partition, spectral backbone decomposition, and departure structure. These invariants are consistent with the coherent dynamics reported in Paper I.

- [Paper II (PDF)](papers/paper-002-h4-invariants/Algebraic_Invariants_of_the_H4_Laplacian_Spectrum.pdf)
- [Paper II (LaTeX source)](papers/paper-002-h4-invariants/paper2_h4_invariants.tex)
- [Spectral figures](papers/paper-002-h4-invariants/figures/)

### Paper III — Nonlinear Mode Localisation on the H₄ Graph
**Focus:** Attractor classification, dynamical taxonomy, and control comparison

The space of persistent nonlinear configurations is classified. Four attractor classes are identified: backbone harmonic modes, phase-locked multi-mode states, spatially localised breather-like configurations, and transitional states. A phase diagram in β delineates the boundaries between linear, weakly nonlinear, and strongly nonlinear regimes. A controlled comparison against random 12-regular graphs and a degree-preserving rewired 600-cell establishes that the structured attractor distribution is non-generic and tied to the H₄ spectral organisation.

- [Paper III (PDF)](papers/paper-003-h4-attractors/Nonlinear_Mode_Localisation_on_the_H4_Graph.pdf)
- [Paper III (LaTeX source)](papers/paper-003-h4-attractors/paper3_h4_attractors.tex)
- [Attractor figures](papers/paper-003-h4-attractors/figures/)
- [Release package](paper-three-release/)

## Relationship

```
Paper I   establishes the phenomenon  (coherent dynamics on H₄ geometry)
Paper II  establishes the structure   (spectral invariants governing the spectrum)
Paper III establishes the taxonomy    (attractor families within the constrained space)
```

Together they indicate that the 600-cell graph possesses a non-generic spectral algebra that produces structured nonlinear dynamics with a classifiable set of recurrent attractor families, as verified against degree-matched control graphs.

## Reproducibility

All simulations and spectral computations are available in [`/simulations`](simulations/). The code includes:

- 600-cell graph construction and verification
- Laplacian eigenvalue computation
- Nonlinear dynamics simulation (symplectic integrator, CPU and GPU)
- Control graph generation and baseline comparison
- Attractor classification and diagnostic pipelines
- Graph-invariant classifier validation

Requirements: Python 3.8+, NumPy, SciPy, Matplotlib. PyTorch with CUDA required for GPU-accelerated Paper III simulations.

## Key Results

| Result | Paper | Value |
|--------|-------|-------|
| Distinct eigenvalues | I, II | 9 (across 120 modes) |
| Localisation enhancement | I | 5× vs controls |
| Shell cross-correlation | I | 0.68 (vs 0.1 controls) |
| Beat frequencies | I | 36 (vs 7000+ controls) |
| Attractor classes | III | 4 (backbone, locked, breather, transitional) |
| Structured trajectories (H₄ vs RRG) | III | 98% vs <4% |
| Localised states (invariant classifier) | III | 44% vs 0.3% |
| Spectral compression (H₄ vs RRG) | III | 9 vs 120 eigenvalues |
| Square multiplicity sequence | II | 1², 2², 3², 4², 5², 6² |
| Spectral backbone | II | 88 = 1 + 3 × 29 |
| φ-cancellation | II | Σb = 0, Σ(b×mult) = 0 |

## Status

This repository contains research-stage work. The results are
mathematically well-defined and computationally verified, but no
physical interpretation is assumed. The spectral structure identified
here is non-generic and warrants further mathematical investigation.

## Author

Lee Smart
Institute of Vibrational Field Dynamics
contact@vibrationalfielddynamics.org
[@vfd_org](https://twitter.com/vfd_org)
