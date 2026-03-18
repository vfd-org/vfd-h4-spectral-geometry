# Paper III — Nonlinear Mode Localisation on the H₄ Graph

Part III of the VFD Research Series on Symmetry-Constrained Resonant Structure.

**[Read the PDF](paper/Nonlinear_Mode_Localisation_on_the_H4_Graph.pdf)**

## Summary

This paper classifies the space of persistent nonlinear dynamical configurations on the 600-cell vertex graph. Four attractor classes are identified — backbone harmonic modes, phase-locked multi-mode states, spatially localised breather-like configurations, and transitional states — and their dependence on the nonlinear coupling strength β is characterised.

A controlled comparison against random 12-regular graphs and a degree-preserving rewired 600-cell establishes that the structured attractor distribution is non-generic and tied to the specific spectral properties of the H₄ graph.

## Contents

```
paper-three-release/
├── README.md                   ← this file
├── paper/
│   ├── paper3_h4_attractors.tex    ← LaTeX source (v1.7, final-hardened)
│   └── Nonlinear_Mode_Localisation_on_the_H4_Graph.pdf
├── figures/
│   ├── fig_phase_diagram.png       ← attractor class distribution vs β
│   ├── fig_ipr_distribution.png    ← IPR histograms across regimes
│   ├── fig_mode_visualisation.png  ← representative configs per class
│   ├── fig_baseline_classes.png    ← H₄ vs RRG class comparison
│   ├── fig_baseline_ipr.png        ← H₄ vs RRG IPR comparison
│   └── fig_baseline_spectrum.png   ← spectral compression (9 vs 120)
├── simulations/
│   ├── build_600cell.py            ← 600-cell graph construction
│   ├── build_control_graphs.py     ← random regular + rewired graph gen
│   ├── laplacian.py                ← Laplacian construction + spectrum
│   ├── integrator.py               ← CPU velocity-Verlet integrator
│   ├── integrator_gpu.py           ← GPU (PyTorch CUDA) integrator
│   ├── diagnostics.py              ← IPR, spectral coefficients, recurrence
│   ├── config.yaml                 ← simulation parameters
│   ├── paper3_gpu_sweep.py         ← main attractor sweep (GPU)
│   ├── paper3_plot_figures.py      ← figure generation from sweep data
│   ├── paper3_baseline_comparison.py ← H₄ vs control graph comparison
│   ├── paper3_invariant_validation.py ← graph-invariant classifier check
│   └── paper3_overnight.sh         ← production run launcher
└── results/
    ├── phase_diagram.npz           ← class fractions vs β (production)
    ├── attractor_classes.npz       ← per-trajectory classifications
    ├── representative_configs.npz  ← example field configurations
    ├── comparison_results.npz      ← H₄ vs 10 RRG baseline data
    └── invariant_results.npz       ← graph-invariant classifier validation
```

## Relationship to Series

```
Paper I   — establishes the phenomenon  (coherent dynamics on H₄ geometry)
Paper II  — establishes the structure   (spectral invariants governing the spectrum)
Paper III — establishes the taxonomy    (attractor families within the constrained space)
```

## Key Results

| Result | Value |
|--------|-------|
| Attractor classes | 4 (backbone, locked, breather, transitional) |
| Distinct eigenvalues (H₄) | 9 |
| Distinct eigenvalues (RRG) | 120 |
| Structured trajectories (H₄) | ~98% |
| Structured trajectories (RRG) | <4% |
| Localised states H₄ vs RRG | 44% vs 0.3% (graph-invariant classifier) |
| Mean IPR ratio (H₄ / RRG) | 2.3× |
| Shannon entropy (H₄ / RRG) | 1.5 bits / 0.2 bits |

## Reproducing Results

### Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib
- PyTorch with CUDA (for GPU-accelerated runs)

### Quick Run (CPU, ~minutes)

```bash
cd simulations
python build_600cell.py          # verify graph construction
python laplacian.py              # compute and display spectrum
```

### Production Attractor Sweep (GPU, ~5 hours)

```bash
cd simulations
python paper3_gpu_sweep.py --n_beta 25 --n_ic 80 --T 2000 --batch 10
python paper3_plot_figures.py --input results/paper3_production
```

### Baseline Comparison (GPU, ~3 hours)

```bash
cd simulations
python paper3_baseline_comparison.py --n_controls 10 --n_ic 40 --n_beta 10 --T 500
```

### Graph-Invariant Validation (GPU, ~2 hours)

```bash
cd simulations
python paper3_invariant_validation.py
```

## Compiling the Paper

```bash
cd paper
pdflatex paper3_h4_attractors.tex
pdflatex paper3_h4_attractors.tex   # second pass for references
```

Figures must be in a `figures/` directory relative to the .tex file, or adjust `\graphicspath` in the preamble.

## Author

Lee Smart
Institute of Vibrational Field Dynamics
contact@vibrationalfielddynamics.org
[@vfd_org](https://twitter.com/vfd_org)

## License

MIT
