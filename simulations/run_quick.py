"""
Quick experiment runner — slim parameter set for fast results.

Runs: 600-cell + 1 control × 2 ICs × 2 betas, T=500.
"""

import os
import json
import numpy as np

from build_600cell import build_600cell
from build_control_graphs import rewired_graph
from laplacian import build_laplacian, compute_spectrum, spectral_summary
from integrator import integrate
from diagnostics import compute_all_diagnostics
from plots import (
    plot_localization_comparison,
    plot_spectral_occupancy,
    plot_example_localization,
    plot_recurrence,
)

RESULTS_DIR = "results"


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(2024)

    # Build graphs
    print("Building 600-cell...")
    A_600, vertices = build_600cell()

    print("Building rewired control...")
    A_ctrl = rewired_graph(A_600, num_swaps=10000, seed=99)

    graphs = {"600cell": A_600, "control1": A_ctrl}

    # Spectra
    print("\nComputing spectra...")
    laplacians = {}
    spectra = {}
    for name, A in graphs.items():
        L = build_laplacian(A)
        evals, evecs = compute_spectrum(L)
        laplacians[name] = L
        spectra[name] = (evals, evecs)
        distinct = spectral_summary(evals)
        print(f"  {name}: {len(distinct)} distinct eigenvalues")

    # Parameters
    omega0 = 1.0
    lam = 1.0
    dt = 0.01
    T = 500
    n_steps = int(T / dt)
    save_interval = 10

    beta_values = [0.0, 0.5]

    all_diag = {}
    all_results = {}

    for beta in beta_values:
        for graph_name, A in graphs.items():
            L = laplacians[graph_name]
            evals, evecs = spectra[graph_name]

            # IC1: single vertex
            Phi0 = np.zeros(120)
            Phi0[0] = 1.0
            Pi0 = np.zeros(120)

            key = f"{graph_name}__IC1__lam{lam}__beta{beta}"
            print(f"  Running {key}...")
            res = integrate(Phi0, Pi0, L, omega0, lam, beta,
                            dt, n_steps, save_interval)
            diag = compute_all_diagnostics(res, evecs, evals)
            all_diag[key] = diag
            all_results[key] = res
            print(f"    drift={diag['energy_drift']:.2e}  "
                  f"IPR_final={diag['ipr'][-1]:.4f}")

            # IC4: local H3 neighbourhood
            Phi0 = np.zeros(120)
            Phi0[0] = 1.0
            neighbors = np.where(A[0] > 0)[0]
            Phi0[neighbors] = 0.5
            Pi0 = np.zeros(120)

            key = f"{graph_name}__IC4__lam{lam}__beta{beta}"
            print(f"  Running {key}...")
            res = integrate(Phi0, Pi0, L, omega0, lam, beta,
                            dt, n_steps, save_interval)
            diag = compute_all_diagnostics(res, evecs, evals)
            all_diag[key] = diag
            all_results[key] = res
            print(f"    drift={diag['energy_drift']:.2e}  "
                  f"IPR_final={diag['ipr'][-1]:.4f}")

    # Plots
    print("\nGenerating plots...")

    # We need the plot functions to handle our subset gracefully.
    # Override the expected keys for the localization plot:
    plot_localization_comparison(all_diag, graphs, RESULTS_DIR)
    plot_spectral_occupancy(all_diag, spectra, RESULTS_DIR)
    plot_example_localization(all_results, vertices, RESULTS_DIR)
    plot_recurrence(all_diag, RESULTS_DIR)

    # Summary
    summary = {
        "600cell_spectrum": [
            {"eigenvalue": float(v), "multiplicity": int(m)}
            for v, m in spectral_summary(spectra["600cell"][0])
        ],
    }
    drifts = {k: float(d["energy_drift"]) for k, d in all_diag.items()}
    summary["energy_drifts"] = drifts

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    run()
