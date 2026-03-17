"""
Experiment runner for VFD nonlinear mode stability simulations.

Runs the full experiment matrix:
  graphs × initial_conditions × parameter sweep
"""

import os
import json
import numpy as np
import yaml

from build_600cell import build_600cell
from build_control_graphs import build_control_graphs
from laplacian import build_laplacian, compute_spectrum, spectral_summary
from integrator import integrate
from diagnostics import compute_all_diagnostics
from plots import (
    plot_localization_comparison,
    plot_spectral_occupancy,
    plot_example_localization,
    plot_recurrence,
)


def make_initial_conditions(n, eigenvectors, A, rng):
    """
    Generate the four initial condition classes.

    Parameters
    ----------
    n : int, number of vertices
    eigenvectors : ndarray (n, n)
    A : ndarray (n, n), adjacency matrix
    rng : numpy random generator

    Returns
    -------
    ics : dict mapping IC name -> (Phi0, Pi0)
    """
    ics = {}

    # IC1: single-vertex excitation
    Phi0 = np.zeros(n)
    Phi0[0] = 1.0
    ics["IC1"] = (Phi0.copy(), np.zeros(n))

    # IC2: random small field
    Phi0 = rng.normal(0, 0.01, n)
    ics["IC2"] = (Phi0.copy(), np.zeros(n))

    # IC3: low-mode seed (second eigenvector)
    Phi0 = eigenvectors[:, 1].copy()
    ics["IC3"] = (Phi0.copy(), np.zeros(n))

    # IC4: local H3 neighbourhood seed
    Phi0 = np.zeros(n)
    v0 = 0
    Phi0[v0] = 1.0
    neighbors = np.where(A[v0] > 0)[0]
    Phi0[neighbors] = 0.5
    ics["IC4"] = (Phi0.copy(), np.zeros(n))

    return ics


def run_single(graph_name, A, L, evals, evecs, ic_name, Phi0, Pi0,
               omega0, lam, beta, dt, n_steps, save_interval):
    """Run a single simulation and compute diagnostics."""
    print(f"  Running: graph={graph_name}, IC={ic_name}, "
          f"λ={lam}, β={beta}")

    results = integrate(
        Phi0, Pi0, L, omega0, lam, beta,
        dt=dt, n_steps=n_steps, save_interval=save_interval
    )

    diag = compute_all_diagnostics(results, evecs, evals)

    print(f"    Energy drift: {diag['energy_drift']:.2e}, "
          f"Final IPR: {diag['ipr'][-1]:.4f}")

    return results, diag


def run_experiment(config_path="config.yaml", results_dir="results"):
    """
    Run the full experiment matrix.

    Parameters
    ----------
    config_path : str, path to config YAML
    results_dir : str, output directory
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dt = config["simulation"]["dt"]
    T = config["simulation"]["T"]
    save_interval = config["simulation"]["save_interval"]
    n_steps = int(T / dt)

    omega0 = config["parameters"]["omega0"]
    lambda_values = config["parameters"]["lambda_values"]
    beta_values = config["parameters"]["beta_values"]

    os.makedirs(results_dir, exist_ok=True)

    rng = np.random.default_rng(2024)

    # Build graphs
    print("=" * 60)
    print("Building 600-cell graph...")
    A_600, vertices = build_600cell()

    print("\nBuilding control graphs...")
    controls = build_control_graphs(A_600)

    graphs = {"600cell": A_600}
    graphs.update(controls)

    # Build Laplacians and spectra
    print("\n" + "=" * 60)
    print("Computing Laplacians and spectra...")
    laplacians = {}
    spectra = {}
    for name, A in graphs.items():
        L = build_laplacian(A)
        evals, evecs = compute_spectrum(L)
        laplacians[name] = L
        spectra[name] = (evals, evecs)

        distinct = spectral_summary(evals)
        print(f"  {name}: {len(distinct)} distinct eigenvalues")

    # Run experiments — focus on key parameter combos
    # Full sweep is large; run a representative subset
    print("\n" + "=" * 60)
    print("Running simulations...")

    all_diagnostics = {}
    all_results = {}

    # Primary comparison: λ=1, sweep β, all ICs on all graphs
    lam = 1.0
    for beta in beta_values:
        for graph_name, A in graphs.items():
            L = laplacians[graph_name]
            evals, evecs = spectra[graph_name]

            ics = make_initial_conditions(120, evecs, A, rng)

            for ic_name in ["IC1", "IC2", "IC3", "IC4"]:
                Phi0, Pi0 = ics[ic_name]

                key = f"{graph_name}__{ic_name}__lam{lam}__beta{beta}"
                results, diag = run_single(
                    graph_name, A, L, evals, evecs,
                    ic_name, Phi0, Pi0,
                    omega0, lam, beta, dt, n_steps, save_interval
                )

                all_diagnostics[key] = diag
                all_results[key] = results

    # Secondary: λ sweep with β=0.5, IC1 only, all graphs
    beta = 0.5
    for lam in lambda_values:
        if lam == 1.0:
            continue  # already done
        for graph_name, A in graphs.items():
            L = laplacians[graph_name]
            evals, evecs = spectra[graph_name]
            ics = make_initial_conditions(120, evecs, A, rng)

            Phi0, Pi0 = ics["IC1"]
            key = f"{graph_name}__IC1__lam{lam}__beta{beta}"
            results, diag = run_single(
                graph_name, A, L, evals, evecs,
                "IC1", Phi0, Pi0,
                omega0, lam, beta, dt, n_steps, save_interval
            )
            all_diagnostics[key] = diag
            all_results[key] = results

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating plots...")

    plot_localization_comparison(
        all_diagnostics, graphs, results_dir
    )
    plot_spectral_occupancy(
        all_diagnostics, spectra, results_dir
    )
    plot_example_localization(
        all_results, vertices, results_dir
    )
    plot_recurrence(
        all_diagnostics, results_dir
    )

    # Save summary
    summary = {
        "graphs": list(graphs.keys()),
        "600cell_spectrum": [
            {"eigenvalue": float(v), "multiplicity": int(m)}
            for v, m in spectral_summary(spectra["600cell"][0])
        ],
    }
    for name in graphs:
        distinct = spectral_summary(spectra[name][0])
        summary[f"{name}_n_distinct_eigenvalues"] = len(distinct)

    energy_drifts = {}
    for key, diag in all_diagnostics.items():
        energy_drifts[key] = float(diag["energy_drift"])
    summary["energy_drifts"] = energy_drifts

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {results_dir}/")
    print("Done.")


if __name__ == "__main__":
    run_experiment()
