"""
Visualization for VFD simulation results.

Produces four figures:
  A - Localization comparison (IPR)
  B - Spectral occupancy heatmap
  C - Example localized state (3D projection)
  D - Recurrence plot
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# Consistent styling
GRAPH_COLORS = {
    "600cell": "#7b2d8e",
    "control1": "#999999",
    "control2": "#bbbbbb",
    "control3": "#ccaa77",
}
GRAPH_LABELS = {
    "600cell": "600-cell ($H_4$)",
    "control1": "Random 12-reg. (1)",
    "control2": "Random 12-reg. (2)",
    "control3": "Rewired 600-cell",
}


def plot_localization_comparison(all_diagnostics, graphs, results_dir):
    """
    Figure A: IPR(t) for 600-cell vs control graphs.

    Plots one panel per β value, IC1, λ=1.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle(
        "Inverse Participation Ratio: 600-cell vs Controls\n"
        "(IC1: single-vertex, $\\lambda=1$)",
        fontsize=13
    )

    beta_values = [0.0, 0.1, 0.5, 1.0]

    for ax, beta in zip(axes.flat, beta_values):
        for graph_name in list(graphs.keys()):
            key = f"{graph_name}__IC1__lam1.0__beta{beta}"
            if key not in all_diagnostics:
                continue
            diag = all_diagnostics[key]
            times = diag["times"]
            ipr = diag["ipr"]

            # Subsample for plotting
            step = max(1, len(times) // 2000)
            ax.plot(
                times[::step], ipr[::step],
                color=GRAPH_COLORS.get(graph_name, "#666666"),
                label=GRAPH_LABELS.get(graph_name, graph_name),
                alpha=0.8, linewidth=0.8
            )

        ax.set_title(f"$\\beta = {beta}$", fontsize=11)
        ax.set_xlabel("Time")
        ax.set_ylabel("IPR")
        ax.set_ylim(0, 1.05)
        ax.axhline(1 / 120, color="gray", linestyle=":", linewidth=0.5,
                    label="Delocalized (1/120)")

    axes[0, 0].legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    path = os.path.join(results_dir, "localization_comparison.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_spectral_occupancy(all_diagnostics, spectra, results_dir):
    """
    Figure B: Spectral sector energy heatmap.

    Shows |c_k(t)|² projected onto eigenmodes, for 600-cell and one control.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Spectral Occupancy: $|c_k(t)|^2$ by Eigenmode Index\n"
        "(IC1, $\\lambda=1$, $\\beta=0.5$)",
        fontsize=13
    )

    for ax, graph_name, title in zip(
        axes,
        ["600cell", "control1"],
        ["600-cell ($H_4$)", "Random 12-regular"]
    ):
        key = f"{graph_name}__IC1__lam1.0__beta0.5"
        if key not in all_diagnostics:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        diag = all_diagnostics[key]
        coeffs = diag["spectral_coefficients"]

        # Sort by eigenvalue order
        evals = spectra[graph_name][0]
        order = np.argsort(evals)
        power = coeffs[:, order] ** 2

        # Subsample time for visualization
        step = max(1, power.shape[0] // 500)
        power_sub = power[::step]
        times_sub = diag["times"][::step]

        vmin = max(power_sub[power_sub > 0].min(), 1e-10)
        im = ax.imshow(
            power_sub.T, aspect="auto",
            origin="lower",
            extent=[times_sub[0], times_sub[-1], 0, 120],
            norm=LogNorm(vmin=vmin, vmax=power_sub.max()),
            cmap="viridis"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Eigenmode index (sorted by $\\mu_k$)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="$|c_k|^2$", shrink=0.8)

    plt.tight_layout()
    path = os.path.join(results_dir, "spectral_occupancy.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_example_localization(all_results, vertices, results_dir):
    """
    Figure C: 3D projection of vertex amplitudes at a late time.

    Shows the field state on the 600-cell projected to 3D.
    """
    key = "600cell__IC1__lam1.0__beta0.5"
    if key not in all_results:
        print("  Skipping example localization plot (no data)")
        return

    results = all_results[key]
    Phi_hist = results["Phi_history"]

    # Take a snapshot at ~80% through the simulation
    t_idx = int(0.8 * len(Phi_hist))
    Phi = Phi_hist[t_idx]

    # Project 4D vertices to 3D (drop last coordinate)
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    # Color and size by |Φ|
    amp = np.abs(Phi)
    amp_norm = amp / (amp.max() + 1e-30)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        x, y, z,
        c=amp_norm, cmap="magma",
        s=10 + 200 * amp_norm**2,
        alpha=0.7, edgecolors="none"
    )

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.set_title(
        f"Field Amplitude on 600-Cell (3D projection)\n"
        f"$t = {results['times'][t_idx]:.0f}$, "
        f"$\\lambda=1$, $\\beta=0.5$, IC1",
        fontsize=11
    )
    fig.colorbar(scatter, ax=ax, label="$|\\Phi_i|$ (normalized)",
                 shrink=0.6)

    plt.tight_layout()
    path = os.path.join(results_dir, "example_localization.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_recurrence(all_diagnostics, results_dir):
    """
    Figure D: Autocorrelation C(τ) for 600-cell vs controls.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left panel: linear regime (β=0)
    # Right panel: nonlinear regime (β=0.5)
    for ax, beta, title in zip(
        axes,
        [0.0, 0.5],
        ["Linear ($\\beta=0$)", "Nonlinear ($\\beta=0.5$)"]
    ):
        for graph_name in ["600cell", "control1", "control2", "control3"]:
            key = f"{graph_name}__IC1__lam1.0__beta{beta}"
            if key not in all_diagnostics:
                continue
            diag = all_diagnostics[key]
            lags = diag["recurrence_lags"]
            corr = diag["recurrence_correlation"]

            # Convert lags to time units
            dt = 0.01
            save_interval = 10
            lag_times = lags * dt * save_interval

            ax.plot(
                lag_times, corr,
                color=GRAPH_COLORS[graph_name],
                label=GRAPH_LABELS[graph_name],
                alpha=0.8, linewidth=1
            )

        ax.set_xlabel("Lag $\\tau$")
        ax.set_ylabel("$C(\\tau)$")
        ax.set_title(f"Recurrence — {title}")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.legend(fontsize=7)

    fig.suptitle(
        "Field Autocorrelation: 600-cell vs Controls (IC1, $\\lambda=1$)",
        fontsize=12
    )
    plt.tight_layout()
    path = os.path.join(results_dir, "recurrence_plot.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path}")
