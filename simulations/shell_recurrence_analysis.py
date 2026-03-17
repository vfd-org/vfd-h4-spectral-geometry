"""
WO-VFD-SIM-004: Shell recurrence and reflection dynamics.

Analyzes whether the 600-cell acts as a finite resonant shell cavity
with coherent wave reflection across distance shells.
"""

import os
import json
import time
import numpy as np
import torch

from build_600cell import build_600cell
from build_control_graphs import rewired_graph, random_regular_graph
from laplacian import build_laplacian, compute_spectrum, spectral_summary
from shell_analysis import compute_graph_distances, shell_partition, shell_energy
from symmetry_tests import detect_breathers

RESULTS_DIR = "results_long"


def integrate_torch(Phi0, Pi0, L_np, omega0, lam, beta, dt, n_steps,
                    save_interval=50):
    """Fast CPU PyTorch integrator."""
    device = torch.device("cpu")
    n = len(Phi0)
    w2 = omega0 ** 2

    Phi = torch.tensor(Phi0, dtype=torch.float64, device=device)
    Pi = torch.tensor(Pi0, dtype=torch.float64, device=device)
    L = torch.tensor(L_np, dtype=torch.float64, device=device)

    n_saves = n_steps // save_interval + 1
    times_np = np.zeros(n_saves)
    Phi_hist = np.zeros((n_saves, n))
    energy_np = np.zeros(n_saves)

    half_dt = 0.5 * dt

    def H():
        k = 0.5 * torch.dot(Pi, Pi)
        p = 0.5 * w2 * torch.dot(Phi, Phi)
        c = 0.5 * lam * torch.dot(Phi, L @ Phi)
        nl = 0.25 * beta * torch.sum(Phi ** 4)
        return (k + p + c + nl).item()

    Phi_hist[0] = Phi.numpy()
    energy_np[0] = H()
    si = 1

    F = -w2 * Phi - lam * (L @ Phi) - beta * Phi * Phi * Phi
    for step in range(1, n_steps + 1):
        Pi.add_(F, alpha=half_dt)
        Phi.add_(Pi, alpha=dt)
        F = -w2 * Phi - lam * (L @ Phi) - beta * Phi * Phi * Phi
        Pi.add_(F, alpha=half_dt)

        if step % save_interval == 0 and si < n_saves:
            times_np[si] = step * dt
            Phi_hist[si] = Phi.numpy()
            energy_np[si] = H()
            si += 1

    return {"times": times_np[:si], "Phi_history": Phi_hist[:si],
            "energy": energy_np[:si]}


def ipr_series(Phi_hist):
    s2 = np.sum(Phi_hist ** 2, axis=1)
    s4 = np.sum(Phi_hist ** 4, axis=1)
    mask = s2 > 1e-30
    out = np.zeros(len(s2))
    out[mask] = s4[mask] / s2[mask] ** 2
    return out


def find_peaks_simple(signal, min_height=None, min_distance=1):
    """Simple peak finder without scipy dependency issues."""
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            if min_height is None or signal[i] >= min_height:
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
    return np.array(peaks)


def cross_correlation(x, y, max_lag=None):
    """Normalized cross-correlation."""
    x = x - np.mean(x)
    y = y - np.mean(y)
    sx = np.std(x)
    sy = np.std(y)
    if sx < 1e-30 or sy < 1e-30:
        return np.zeros(1), np.array([0])

    if max_lag is None:
        max_lag = len(x) // 4

    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.zeros(len(lags))
    n = len(x)

    for i, lag in enumerate(lags):
        if lag >= 0:
            corr[i] = np.dot(x[:n - lag], y[lag:]) / (n * sx * sy)
        else:
            corr[i] = np.dot(x[-lag:], y[:n + lag]) / (n * sx * sy)

    return corr, lags


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("WO-VFD-SIM-004: Shell Recurrence & Reflection Analysis")
    print("=" * 60)

    # Build graphs
    print("\nBuilding graphs...")
    A_600, vertices = build_600cell()
    A_ctrl1 = rewired_graph(A_600, num_swaps=10000, seed=99)
    A_ctrl2 = random_regular_graph(120, 12, seed=42)

    graphs = {"600cell": A_600, "control1": A_ctrl1, "control2": A_ctrl2}

    # Compute Laplacians
    laplacians = {}
    for name, A in graphs.items():
        laplacians[name] = build_laplacian(A).astype(np.float64)

    # Shell structures
    print("\nShell structures:")
    shell_data = {}
    for name, A in graphs.items():
        dists = compute_graph_distances(A, 0)
        shells = shell_partition(dists)
        shell_data[name] = {"distances": dists, "shells": shells}
        sizes = [len(shells[d]) for d in sorted(shells)]
        print(f"  {name}: {sizes}")

    # Parameters
    omega0, lam, dt = 1.0, 1.0, 0.01
    T = 500
    n_steps = int(T / dt)
    save_interval = 20  # finer resolution for recurrence analysis
    betas = [0.0, 0.5]

    all_data = {}

    # Run simulations
    print(f"\nRunning simulations (T={T}, PyTorch CPU)...")
    for beta in betas:
        for gname in graphs:
            Phi0 = np.zeros(120); Phi0[0] = 1.0
            key = f"{gname}__beta{beta}"
            print(f"  {key}...", end=" ", flush=True)
            t0 = time.time()
            res = integrate_torch(Phi0, np.zeros(120), laplacians[gname],
                                  omega0, lam, beta, dt, n_steps,
                                  save_interval)
            elapsed = time.time() - t0
            ipr = ipr_series(res["Phi_history"])

            # Shell energy
            shells = shell_data[gname]["shells"]
            sE = shell_energy(res["Phi_history"], shells)

            all_data[key] = {"res": res, "ipr": ipr, "shell_energy": sE}
            print(f"{elapsed:.1f}s  IPR_mean={ipr.mean():.4f}")

    # ── Analysis ─────────────────────────────────────────────
    print("\nAnalyzing shell recurrence...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = {"600cell": "#7b2d8e", "control1": "#999999", "control2": "#bbbbbb"}
    LABELS = {"600cell": "600-cell ($H_4$)", "control1": "Rewired ctrl",
              "control2": "Random 12-reg"}

    reflection_data = {}

    for beta in betas:
        for gname in graphs:
            key = f"{gname}__beta{beta}"
            sE = all_data[key]["shell_energy"]
            t = all_data[key]["res"]["times"]

            # E0 peaks (origin shell recurrence)
            E0 = sE[0]
            peaks = find_peaks_simple(E0, min_height=np.mean(E0) + np.std(E0),
                                      min_distance=3)

            if len(peaks) > 1:
                intervals = np.diff(t[peaks])
                mean_period = float(np.mean(intervals))
                std_period = float(np.std(intervals))
            else:
                intervals = np.array([])
                mean_period = 0.0
                std_period = 0.0

            # Cross-correlation E0 vs outermost shell
            max_shell = max(sE.keys())
            E_outer = sE[max_shell]
            max_lag = min(len(t) // 4, 200)
            corr_0_outer, lags_0_outer = cross_correlation(E0, E_outer,
                                                            max_lag)

            # Find reflection delay
            if len(corr_0_outer) > 0:
                peak_lag_idx = np.argmax(corr_0_outer)
                reflect_lag = float(lags_0_outer[peak_lag_idx])
                reflect_time = reflect_lag * dt * save_interval
                peak_corr = float(corr_0_outer[peak_lag_idx])
            else:
                reflect_time = 0.0
                peak_corr = 0.0

            reflection_data[key] = {
                "n_peaks": int(len(peaks)),
                "mean_period": mean_period,
                "std_period": std_period,
                "reflection_delay": reflect_time,
                "peak_correlation": peak_corr,
            }
            print(f"  {key}: {len(peaks)} recurrence peaks, "
                  f"period={mean_period:.2f}±{std_period:.2f}, "
                  f"reflect_delay={reflect_time:.2f}, "
                  f"corr={peak_corr:.3f}")

    # ── Figure 1: Shell reflection map ───────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Shell Energy Propagation Maps", fontsize=14)

    plot_configs = [
        ("600cell", 0.0), ("600cell", 0.5),
        ("control1", 0.0), ("control1", 0.5),
    ]
    for ax, (gname, beta) in zip(axes.flat, plot_configs):
        key = f"{gname}__beta{beta}"
        sE = all_data[key]["shell_energy"]
        t = all_data[key]["res"]["times"]

        n_shells = len(sE)
        map_data = np.zeros((n_shells, len(t)))
        for d in range(n_shells):
            if d in sE:
                map_data[d] = sE[d]

        # Normalize each row
        for d in range(n_shells):
            mx = map_data[d].max()
            if mx > 1e-30:
                map_data[d] /= mx

        im = ax.imshow(map_data, aspect="auto", origin="lower",
                       extent=[t[0], t[-1], -0.5, n_shells - 0.5],
                       cmap="inferno", interpolation="nearest")
        ax.set_xlabel("Time"); ax.set_ylabel("Distance shell $d$")
        ax.set_title(f"{LABELS.get(gname, gname)}, $\\beta={beta}$")
        fig.colorbar(im, ax=ax, label="Normalized energy", shrink=0.8)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "shell_reflection_map.png"),
                dpi=200)
    plt.close(fig)
    print("  Saved shell_reflection_map.png")

    # ── Figure 2: Recurrence period ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, beta in zip(axes, betas):
        for gname in graphs:
            key = f"{gname}__beta{beta}"
            sE = all_data[key]["shell_energy"]
            t = all_data[key]["res"]["times"]
            E0 = sE[0]
            step = max(1, len(t) // 3000)
            ax.plot(t[::step], E0[::step], color=COLORS[gname],
                    label=LABELS[gname], alpha=0.7, lw=0.7)
        ax.set_xlabel("Time")
        ax.set_ylabel("Origin shell energy $E_0(t)$")
        ax.set_title(f"Origin Recurrence ($\\beta={beta}$)")
        ax.legend(fontsize=7)
    fig.suptitle("Shell Recurrence at Origin Vertex", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "shell_recurrence_period.png"),
                dpi=200)
    plt.close(fig)
    print("  Saved shell_recurrence_period.png")

    # ── Figure 3: Shell phase correlation ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Shell Cross-Correlation (E₀ vs E_outer)", fontsize=13)

    for ax, beta in zip(axes, betas):
        for gname in graphs:
            key = f"{gname}__beta{beta}"
            sE = all_data[key]["shell_energy"]
            t = all_data[key]["res"]["times"]
            max_shell = max(sE.keys())
            E0 = sE[0]
            E_out = sE[max_shell]
            max_lag = min(len(t) // 4, 200)
            corr, lags = cross_correlation(E0, E_out, max_lag)
            lag_times = lags * dt * save_interval

            ax.plot(lag_times, corr, color=COLORS[gname],
                    label=LABELS[gname], alpha=0.8, lw=1)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("Lag (time units)")
        ax.set_ylabel("Cross-correlation")
        ax.set_title(f"$\\beta={beta}$")
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "shell_phase_correlation.png"),
                dpi=200)
    plt.close(fig)
    print("  Saved shell_phase_correlation.png")

    # ── Save reflection data ─────────────────────────────────
    with open(os.path.join(RESULTS_DIR, "reflection_delay.json"), "w") as f:
        json.dump(reflection_data, f, indent=2)
    print("  Saved reflection_delay.json")

    # ── Summary report ───────────────────────────────────────
    lines = [
        "# WO-VFD-SIM-004: Shell Recurrence Summary\n",
        "## Recurrence Statistics\n",
        "| Run | Peaks | Period (mean±std) | Reflect delay | Corr |",
        "|-----|-------|-------------------|---------------|------|",
    ]
    for beta in betas:
        for gname in graphs:
            key = f"{gname}__beta{beta}"
            rd = reflection_data[key]
            lines.append(
                f"| {key} | {rd['n_peaks']} | "
                f"{rd['mean_period']:.2f}±{rd['std_period']:.2f} | "
                f"{rd['reflection_delay']:.2f} | "
                f"{rd['peak_correlation']:.3f} |"
            )

    lines.extend([
        "\n## Shell Sizes\n",
        "- 600-cell: 1, 12, 32, 42, 32, 1 (diameter 5)",
    ])
    for gname in ["control1", "control2"]:
        shells = shell_data[gname]["shells"]
        sizes = [len(shells[d]) for d in sorted(shells)]
        lines.append(f"- {gname}: {sizes}")

    with open(os.path.join(RESULTS_DIR, "shell_recurrence_summary.md"),
              "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  Saved shell_recurrence_summary.md")

    print("\nDone!")


if __name__ == "__main__":
    run()
