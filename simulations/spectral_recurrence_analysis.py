"""
WO-VFD-SIM-005: Spectral Gap and Recurrence Analysis.

Tests whether IPR recurrence periods match beat frequencies
between the 9 distinct Laplacian eigenvalue sectors.
"""

import os
import json
import time
import numpy as np
import torch

from build_600cell import build_600cell
from build_control_graphs import rewired_graph, random_regular_graph
from laplacian import build_laplacian, compute_spectrum, spectral_summary

RESULTS_DIR = "results_long"


def integrate_torch(Phi0, Pi0, L_np, omega0, lam, beta, dt, n_steps,
                    save_interval=20):
    device = torch.device("cpu")
    n = len(Phi0)
    w2 = omega0 ** 2
    Phi = torch.tensor(Phi0, dtype=torch.float64, device=device)
    Pi = torch.tensor(Pi0, dtype=torch.float64, device=device)
    L = torch.tensor(L_np, dtype=torch.float64, device=device)

    n_saves = n_steps // save_interval + 1
    times = np.zeros(n_saves)
    Phi_hist = np.zeros((n_saves, n))
    energy = np.zeros(n_saves)
    half_dt = 0.5 * dt

    def H():
        return (0.5 * torch.dot(Pi, Pi) + 0.5 * w2 * torch.dot(Phi, Phi)
                + 0.5 * lam * torch.dot(Phi, L @ Phi)
                + 0.25 * beta * torch.sum(Phi ** 4)).item()

    Phi_hist[0] = Phi.numpy()
    energy[0] = H()
    si = 1
    F = -w2 * Phi - lam * (L @ Phi) - beta * Phi * Phi * Phi
    for step in range(1, n_steps + 1):
        Pi.add_(F, alpha=half_dt)
        Phi.add_(Pi, alpha=dt)
        F = -w2 * Phi - lam * (L @ Phi) - beta * Phi * Phi * Phi
        Pi.add_(F, alpha=half_dt)
        if step % save_interval == 0 and si < n_saves:
            times[si] = step * dt
            Phi_hist[si] = Phi.numpy()
            energy[si] = H()
            si += 1
    return {"times": times[:si], "Phi_history": Phi_hist[:si],
            "energy": energy[:si]}


def ipr_series(Phi_hist):
    s2 = np.sum(Phi_hist ** 2, axis=1)
    s4 = np.sum(Phi_hist ** 4, axis=1)
    mask = s2 > 1e-30
    out = np.zeros(len(s2))
    out[mask] = s4[mask] / s2[mask] ** 2
    return out


def find_peaks(signal, min_height=None, min_distance=1):
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            if min_height is None or signal[i] >= min_height:
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
    return np.array(peaks)


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("WO-VFD-SIM-005: Spectral Gap & Recurrence Analysis")
    print("=" * 60)

    # Build graphs
    print("\nBuilding graphs...")
    A_600, _ = build_600cell()
    A_ctrl = rewired_graph(A_600, num_swaps=10000, seed=99)

    graphs = {"600cell": A_600, "control": A_ctrl}
    laplacians = {}
    spectra = {}
    for name, A in graphs.items():
        L = build_laplacian(A).astype(np.float64)
        evals, evecs = compute_spectrum(L)
        laplacians[name] = L
        spectra[name] = (evals, evecs)

    # Parameters
    omega0, lam, beta, dt = 1.0, 1.0, 0.5, 0.01
    T = 500
    n_steps = int(T / dt)
    save_interval = 10  # fine resolution for frequency analysis

    # ── Task 1: Mode frequencies ─────────────────────────────
    print("\nTask 1: Computing mode frequencies...")
    sector_freqs = {}
    for name in graphs:
        evals = spectra[name][0]
        distinct = spectral_summary(evals)
        omegas = [np.sqrt(omega0 ** 2 + lam * mu) for mu, _ in distinct]
        sector_freqs[name] = {
            "eigenvalues": [float(mu) for mu, _ in distinct],
            "multiplicities": [int(m) for _, m in distinct],
            "Omega": omegas,
        }
        print(f"  {name}: {len(distinct)} sectors")
        for i, (mu, m) in enumerate(distinct):
            print(f"    μ={mu:.4f} (×{m})  Ω={omegas[i]:.4f}")

    # ── Task 2: Frequency gap matrix ────────────────────────
    print("\nTask 2: Computing gap matrices...")
    gap_data = {}
    for name in graphs:
        Om = np.array(sector_freqs[name]["Omega"])
        n_sec = len(Om)
        gaps = np.abs(Om[:, None] - Om[None, :])
        gap_data[name] = gaps

    # ── Task 3: Predicted recurrence periods ─────────────────
    print("\nTask 3: Predicting recurrence periods...")
    predicted = {}
    for name in graphs:
        gaps = gap_data[name]
        n_sec = gaps.shape[0]
        periods = []
        for i in range(n_sec):
            for j in range(i + 1, n_sec):
                if gaps[i, j] > 1e-8:
                    T_beat = 2 * np.pi / gaps[i, j]
                    mi = sector_freqs[name]["multiplicities"][i]
                    mj = sector_freqs[name]["multiplicities"][j]
                    weight = mi * mj  # sector size product
                    periods.append({
                        "sectors": (i, j),
                        "gap": float(gaps[i, j]),
                        "period": float(T_beat),
                        "weight": int(weight),
                    })
        periods.sort(key=lambda x: x["weight"], reverse=True)
        predicted[name] = periods
        print(f"  {name}: {len(periods)} beat periods")
        for p in periods[:5]:
            print(f"    sectors {p['sectors']}: Δ={p['gap']:.4f}, "
                  f"T={p['period']:.4f}, weight={p['weight']}")

    # ── Task 4: Simulate and measure IPR recurrence ──────────
    print(f"\nTask 4: Running simulations (T={T})...")
    sim_data = {}
    for name in graphs:
        Phi0 = np.zeros(120); Phi0[0] = 1.0
        print(f"  {name}...", end=" ", flush=True)
        t0 = time.time()
        res = integrate_torch(Phi0, np.zeros(120), laplacians[name],
                              omega0, lam, beta, dt, n_steps, save_interval)
        elapsed = time.time() - t0
        ipr = ipr_series(res["Phi_history"])
        sim_data[name] = {"res": res, "ipr": ipr}
        print(f"{elapsed:.1f}s  IPR_mean={ipr.mean():.4f}")

    # Detect IPR peaks
    observed_intervals = {}
    for name in graphs:
        ipr = sim_data[name]["ipr"]
        t = sim_data[name]["res"]["times"]
        threshold = np.mean(ipr) + 1.5 * np.std(ipr)
        peaks = find_peaks(ipr, min_height=threshold, min_distance=3)
        if len(peaks) > 1:
            intervals = np.diff(t[peaks])
        else:
            intervals = np.array([])
        observed_intervals[name] = intervals
        print(f"  {name}: {len(peaks)} IPR peaks, "
              f"mean interval={intervals.mean():.3f}" if len(intervals) > 0
              else f"  {name}: {len(peaks)} IPR peaks")

    # ── Task 5: IPR power spectrum for direct frequency comparison ──
    print("\nComputing IPR power spectra...")
    ipr_spectra = {}
    for name in graphs:
        ipr = sim_data[name]["ipr"]
        t = sim_data[name]["res"]["times"]
        dt_save = t[1] - t[0]

        # Remove mean, window, FFT
        ipr_centered = ipr - np.mean(ipr)
        window = np.hanning(len(ipr_centered))
        ipr_windowed = ipr_centered * window
        fft = np.fft.rfft(ipr_windowed)
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(ipr_windowed), d=dt_save)

        ipr_spectra[name] = {"freqs": freqs, "power": power}

    # ── Plots ────────────────────────────────────────────────
    print("\nGenerating plots...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: Spectral gap matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Frequency Gap Matrix $|\\Omega_i - \\Omega_j|$", fontsize=13)
    for ax, name, title in zip(axes, ["600cell", "control"],
                               ["600-cell (9 sectors)", "Control (120 sectors)"]):
        gaps = gap_data[name]
        n_sec = gaps.shape[0]
        if n_sec > 30:
            # Show only first 30 for control
            gaps = gaps[:30, :30]
        im = ax.imshow(gaps, cmap="viridis", origin="lower")
        ax.set_xlabel("Sector $j$"); ax.set_ylabel("Sector $i$")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="$|\\Omega_i - \\Omega_j|$", shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "spectral_gap_matrix.png"), dpi=200)
    plt.close(fig)
    print("  Saved spectral_gap_matrix.png")

    # Figure 2: Predicted recurrence periods
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Predicted Beat Periods (weighted by sector size)", fontsize=13)
    for ax, name, title in zip(axes, ["600cell", "control"],
                               ["600-cell", "Control"]):
        preds = predicted[name]
        periods = [p["period"] for p in preds]
        weights = [p["weight"] for p in preds]
        if periods:
            ax.stem(periods[:30], weights[:30], linefmt="C0-",
                    markerfmt="C0o", basefmt="gray")
        ax.set_xlabel("Beat period $T_{ij}$")
        ax.set_ylabel("Weight ($m_i \\times m_j$)")
        ax.set_title(title)
        ax.set_xlim(0, min(50, max(periods[:30]) * 1.1) if periods else 50)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "recurrence_period_prediction.png"),
                dpi=200)
    plt.close(fig)
    print("  Saved recurrence_period_prediction.png")

    # Figure 3: Observed recurrence histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Observed IPR Recurrence Intervals", fontsize=13)
    for ax, name, title, color in zip(axes, ["600cell", "control"],
                                       ["600-cell", "Control"],
                                       ["#7b2d8e", "#999999"]):
        intervals = observed_intervals[name]
        if len(intervals) > 0:
            ax.hist(intervals, bins=30, color=color, alpha=0.7, density=True)
            ax.axvline(np.mean(intervals), color="red", ls="--", lw=1.5,
                       label=f"mean={np.mean(intervals):.2f}")
            # Overlay predicted dominant periods
            if name == "600cell":
                top_preds = sorted(predicted[name],
                                   key=lambda x: x["weight"], reverse=True)[:5]
                for p in top_preds:
                    ax.axvline(p["period"], color="blue", ls=":", lw=1,
                               alpha=0.5)
                ax.axvline(0, color="blue", ls=":", lw=1, alpha=0.5,
                           label="Predicted beats")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "Insufficient peaks", ha="center",
                    transform=ax.transAxes)
        ax.set_xlabel("Recurrence interval")
        ax.set_ylabel("Density")
        ax.set_title(title)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "observed_recurrence_histogram.png"),
                dpi=200)
    plt.close(fig)
    print("  Saved observed_recurrence_histogram.png")

    # Figure 4: IPR power spectrum vs predicted beat frequencies
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("IPR Power Spectrum vs Predicted Beat Frequencies", fontsize=13)
    for ax, name, title, color in zip(axes, ["600cell", "control"],
                                       ["600-cell", "Control"],
                                       ["#7b2d8e", "#999999"]):
        sp = ipr_spectra[name]
        freqs = sp["freqs"]
        power = sp["power"]
        # Convert to angular frequency for comparison
        omega_freqs = 2 * np.pi * freqs

        # Normalize power
        power_norm = power / (power.max() + 1e-30)

        ax.plot(omega_freqs, power_norm, color=color, lw=0.7, alpha=0.8)

        # Overlay predicted gaps
        if name == "600cell":
            gaps_flat = gap_data[name][np.triu_indices(gap_data[name].shape[0],
                                                       k=1)]
            gaps_flat = gaps_flat[gaps_flat > 0.01]
            for g in gaps_flat:
                ax.axvline(g, color="red", ls=":", lw=0.5, alpha=0.3)
            ax.axvline(0, color="red", ls=":", lw=0.5, alpha=0.3,
                       label="Predicted gaps $\\Delta_{ij}$")
            ax.legend(fontsize=8)

        ax.set_xlabel("Angular frequency $\\omega$")
        ax.set_ylabel("Normalized IPR power")
        ax.set_title(title)
        ax.set_xlim(0, 6)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "ipr_recurrence_vs_spectrum.png"),
                dpi=200)
    plt.close(fig)
    print("  Saved ipr_recurrence_vs_spectrum.png")

    # Figure 5: Gap density comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, color, label in [("600cell", "#7b2d8e", "600-cell (9 sectors)"),
                                ("control", "#999999", "Control (120 sectors)")]:
        gaps = gap_data[name]
        gaps_flat = gaps[np.triu_indices(gaps.shape[0], k=1)]
        gaps_flat = gaps_flat[gaps_flat > 0.01]
        ax.hist(gaps_flat, bins=50, color=color, alpha=0.5, density=True,
                label=label)
    ax.set_xlabel("Frequency gap $\\Delta_{ij}$")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Frequency Gaps")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "gap_density_comparison.png"),
                dpi=200)
    plt.close(fig)
    print("  Saved gap_density_comparison.png")

    # ── Summary ──────────────────────────────────────────────
    lines = [
        "# WO-VFD-SIM-005: Spectral Gap & Recurrence Summary\n",
        f"**Parameters:** ω₀={omega0}, λ={lam}, β={beta}, T={T}\n",
        "\n## Mode Frequencies (600-cell)\n",
        "| Sector | μ_k | Ω_k | Multiplicity |",
        "|--------|-----|-----|-------------|",
    ]
    sf = sector_freqs["600cell"]
    for i in range(len(sf["eigenvalues"])):
        lines.append(f"| {i+1} | {sf['eigenvalues'][i]:.4f} | "
                     f"{sf['Omega'][i]:.4f} | {sf['multiplicities'][i]} |")

    lines.extend([
        "\n## Top 10 Predicted Beat Periods (600-cell)\n",
        "| Sectors | Gap | Period | Weight |",
        "|---------|-----|--------|--------|",
    ])
    for p in predicted["600cell"][:10]:
        lines.append(f"| {p['sectors']} | {p['gap']:.4f} | "
                     f"{p['period']:.4f} | {p['weight']} |")

    lines.extend([
        "\n## Observed IPR Recurrence\n",
    ])
    for name in graphs:
        intervals = observed_intervals[name]
        if len(intervals) > 0:
            lines.append(f"- **{name}**: {len(intervals)} intervals, "
                         f"mean={np.mean(intervals):.3f}, "
                         f"std={np.std(intervals):.3f}, "
                         f"median={np.median(intervals):.3f}")
        else:
            lines.append(f"- **{name}**: insufficient peaks")

    lines.extend([
        "\n## Gap Density\n",
        f"- 600-cell: {len(predicted['600cell'])} unique gaps "
        f"from 9 sectors",
        f"- Control: {len(predicted['control'])} unique gaps "
        f"from {len(sector_freqs['control']['Omega'])} sectors",
    ])

    with open(os.path.join(RESULTS_DIR, "spectral_gap_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  Saved spectral_gap_summary.md")

    # Save predicted periods
    with open(os.path.join(RESULTS_DIR, "predicted_periods.json"), "w") as f:
        json.dump({"600cell": predicted["600cell"][:20],
                   "control": predicted["control"][:20]}, f, indent=2)
    print("  Saved predicted_periods.json")

    print("\nDone!")


if __name__ == "__main__":
    run()
