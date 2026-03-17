"""
WO-VFD-SIM-003: Isospectral surrogate control experiment.

Tests whether observed localization depends on H₄ eigenvector geometry
or merely on spectral compression.

Compares:
  - 600-cell (original)
  - Full isospectral surrogate (random orthogonal basis)
  - Block isospectral surrogate (randomized within sectors)
  - Degree-matched control (rewired, from previous experiments)
"""

import os
import json
import time
import numpy as np
from scipy import sparse

from build_600cell import build_600cell
from build_control_graphs import rewired_graph
from laplacian import build_laplacian, compute_spectrum, spectral_summary
from isospectral_control import (build_full_surrogate, build_block_surrogate,
                                  verify_surrogate)
from symmetry_tests import detect_breathers

RESULTS_DIR = "results_isospectral"


def integrate_matrix(Phi0, Pi0, L, omega0, lam, beta, dt, n_steps,
                     save_interval=100):
    """Velocity-Verlet integrator accepting dense matrix L."""
    Phi = Phi0.astype(np.float64).copy()
    Pi = Pi0.astype(np.float64).copy()
    n = len(Phi)
    w2 = omega0**2

    # Use sparse if possible, dense otherwise
    if sparse.issparse(L):
        Ldot = lambda x: L.dot(x)
    else:
        L_dense = np.ascontiguousarray(L, dtype=np.float64)
        Ldot = lambda x: L_dense @ x

    n_saves = n_steps // save_interval + 1
    times = np.zeros(n_saves)
    Phi_hist = np.zeros((n_saves, n))
    energy = np.zeros(n_saves)

    def H():
        return (0.5 * np.dot(Pi, Pi)
                + 0.5 * w2 * np.dot(Phi, Phi)
                + 0.5 * lam * Phi.dot(Ldot(Phi))
                + 0.25 * beta * np.sum(Phi**4))

    Phi_hist[0] = Phi
    energy[0] = H()
    si = 1

    F = -w2 * Phi - lam * Ldot(Phi) - beta * Phi * Phi * Phi
    for step in range(1, n_steps + 1):
        Pi += 0.5 * dt * F
        Phi += dt * Pi
        F = -w2 * Phi - lam * Ldot(Phi) - beta * Phi * Phi * Phi
        Pi += 0.5 * dt * F

        if step % save_interval == 0 and si < n_saves:
            times[si] = step * dt
            Phi_hist[si] = Phi
            energy[si] = H()
            si += 1

    return {"times": times[:si], "Phi_history": Phi_hist[:si],
            "energy": energy[:si]}


def ipr_series(Phi_hist):
    s2 = np.sum(Phi_hist**2, axis=1)
    s4 = np.sum(Phi_hist**4, axis=1)
    mask = s2 > 1e-30
    out = np.zeros(len(s2))
    out[mask] = s4[mask] / s2[mask]**2
    return out


def spectral_entropy_from_L(Phi_hist, L_matrix):
    """Compute spectral entropy using eigenbasis of given Laplacian."""
    evals, evecs = np.linalg.eigh(L_matrix)
    coeffs = Phi_hist @ evecs

    sorted_idx = np.argsort(evals)
    sorted_ev = evals[sorted_idx]

    sectors = []
    i = 0
    while i < len(sorted_ev):
        j = i + 1
        while j < len(sorted_ev) and abs(sorted_ev[j] - sorted_ev[i]) < 1e-6:
            j += 1
        sectors.append(sorted_idx[i:j])
        i = j

    n_times = coeffs.shape[0]
    entropy = np.zeros(n_times)

    for t in range(n_times):
        sector_E = []
        for indices in sectors:
            sector_E.append(np.sum(coeffs[t, indices]**2))
        sector_E = np.array(sector_E)
        total = sector_E.sum()
        if total < 1e-30:
            continue
        p = sector_E / total
        p = p[p > 1e-30]
        entropy[t] = -np.sum(p * np.log(p))

    return entropy


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("WO-VFD-SIM-003: Isospectral Surrogate Control Experiment")
    print("=" * 60)

    # Build 600-cell
    print("\nBuilding 600-cell...")
    A_600, vertices = build_600cell()
    L_600 = build_laplacian(A_600).astype(np.float64)
    evals_600, evecs_600 = compute_spectrum(L_600)

    print(f"  600-cell: {len(spectral_summary(evals_600))} distinct eigenvalues")

    # Build surrogates
    print("\nBuilding isospectral surrogates...")
    L_full, Q_full = build_full_surrogate(evals_600, seed=42)
    verify_surrogate(L_600, L_full, "full_surrogate")

    L_block, Q_block = build_block_surrogate(evals_600, evecs_600, seed=42)
    verify_surrogate(L_600, L_block, "block_surrogate")

    # Rewired control (for reference)
    print("\nBuilding rewired control...")
    A_ctrl = rewired_graph(A_600, num_swaps=10000, seed=99)
    L_ctrl = build_laplacian(A_ctrl).astype(np.float64)

    # Systems to compare
    systems = {
        "600cell": L_600,
        "iso_full": L_full,
        "iso_block": L_block,
        "rewired": L_ctrl,
    }

    COLORS = {
        "600cell": "#7b2d8e",
        "iso_full": "#c44e52",
        "iso_block": "#e07a3a",
        "rewired": "#999999",
    }
    LABELS = {
        "600cell": "600-cell ($H_4$)",
        "iso_full": "Isospectral (full rand.)",
        "iso_block": "Isospectral (block rand.)",
        "rewired": "Rewired control",
    }

    # Parameters
    omega0, lam, dt = 1.0, 1.0, 0.01
    T = 500
    n_steps = int(T / dt)
    save_interval = 50
    betas = [0.0, 0.5, 1.0]

    all_data = {}

    # Run simulations
    print(f"\nRunning simulations (T={T})...")
    for beta in betas:
        for sname, L in systems.items():
            # IC1: single-site kick
            Phi0 = np.zeros(120); Phi0[0] = 1.0

            key = f"{sname}__IC1__beta{beta}"
            print(f"  {key}...", end=" ", flush=True)
            t0 = time.time()
            res = integrate_matrix(Phi0, np.zeros(120), L,
                                   omega0, lam, beta, dt, n_steps,
                                   save_interval)
            elapsed = time.time() - t0
            ipr = ipr_series(res["Phi_history"])
            ent = spectral_entropy_from_L(res["Phi_history"], L)
            drift = abs(res["energy"][-1] - res["energy"][0]) / \
                    (abs(res["energy"][0]) + 1e-30)
            breathers = detect_breathers(ipr, res["times"],
                                         threshold=0.3, min_duration_steps=3)
            all_data[key] = {
                "res": res, "ipr": ipr, "entropy": ent,
                "breather_count": len(breathers),
            }
            print(f"{elapsed:.1f}s  drift={drift:.2e}  "
                  f"IPR={ipr.mean():.4f}  ent={ent.mean():.2f}  "
                  f"breathers={len(breathers)}")

    # ── Plots ────────────────────────────────────────────────
    print("\nGenerating plots...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: IPR comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    fig.suptitle("IPR: 600-cell vs Isospectral Surrogates (IC1)", fontsize=13)
    for ax, beta in zip(axes, betas):
        for sname in systems:
            key = f"{sname}__IC1__beta{beta}"
            d = all_data[key]
            t = d["res"]["times"]
            step = max(1, len(t) // 2000)
            ax.plot(t[::step], d["ipr"][::step],
                    color=COLORS[sname], label=LABELS[sname],
                    alpha=0.7, lw=0.7)
        ax.axhline(1/120, color="gray", ls=":", lw=0.5)
        ax.set_title(f"$\\beta={beta}$"); ax.set_xlabel("Time")
        ax.set_ylabel("IPR"); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "ipr_isospectral_comparison.png"),
                dpi=200)
    plt.close(fig)
    print("  Saved ipr_isospectral_comparison.png")

    # Figure 2: Spectral entropy comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    fig.suptitle("Spectral Entropy: 600-cell vs Isospectral Surrogates (IC1)",
                 fontsize=13)
    for ax, beta in zip(axes, betas):
        for sname in systems:
            key = f"{sname}__IC1__beta{beta}"
            d = all_data[key]
            t = d["res"]["times"]
            step = max(1, len(t) // 2000)
            ax.plot(t[::step], d["entropy"][::step],
                    color=COLORS[sname], label=LABELS[sname],
                    alpha=0.7, lw=0.8)
        ax.set_title(f"$\\beta={beta}$"); ax.set_xlabel("Time")
        ax.set_ylabel("Spectral entropy")
        ax.legend(fontsize=6)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR,
                             "spectral_entropy_isospectral.png"), dpi=200)
    plt.close(fig)
    print("  Saved spectral_entropy_isospectral.png")

    # Figure 3: Breather count bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = []
    x_pos = []
    colors_bars = []
    heights = []
    pos = 0
    for beta in betas:
        for sname in systems:
            key = f"{sname}__IC1__beta{beta}"
            x_labels.append(f"{LABELS[sname]}\nβ={beta}")
            x_pos.append(pos)
            colors_bars.append(COLORS[sname])
            heights.append(all_data[key]["breather_count"])
            pos += 1
        pos += 0.5  # gap between beta groups

    ax.bar(x_pos, heights, color=colors_bars, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Breather events (IPR > 0.3)")
    ax.set_title("Breather-like Localisation Events")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "breather_counts_isospectral.png"),
                dpi=200)
    plt.close(fig)
    print("  Saved breather_counts_isospectral.png")

    # Figure 4: Time trace overlay (β=0.5)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    beta = 0.5
    for sname in systems:
        key = f"{sname}__IC1__beta{beta}"
        d = all_data[key]
        t = d["res"]["times"]
        step = max(1, len(t) // 3000)
        ax.plot(t[::step], d["ipr"][::step],
                color=COLORS[sname], label=LABELS[sname],
                alpha=0.7, lw=0.7)
    ax.axhline(1/120, color="gray", ls=":", lw=0.5)
    ax.set_xlabel("Time"); ax.set_ylabel("IPR")
    ax.set_title(f"IPR Time Trace Comparison ($\\beta={beta}$, IC1)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "time_trace_isospectral.png"),
                dpi=200)
    plt.close(fig)
    print("  Saved time_trace_isospectral.png")

    # ── Summary ──────────────────────────────────────────────
    lines = [
        "# WO-VFD-SIM-003: Isospectral Surrogate Results\n",
        f"**Parameters:** ω₀={omega0}, λ={lam}, dt={dt}, T={T}\n",
        "\n## Mean IPR (IC1)\n",
        "| System | β=0 | β=0.5 | β=1.0 |",
        "|--------|-----|-------|-------|",
    ]
    for sname in systems:
        vals = []
        for beta in betas:
            key = f"{sname}__IC1__beta{beta}"
            vals.append(f"{all_data[key]['ipr'].mean():.4f}")
        lines.append(f"| {LABELS[sname]} | {' | '.join(vals)} |")

    lines.extend([
        "\n## Mean Spectral Entropy (IC1)\n",
        "| System | β=0 | β=0.5 | β=1.0 |",
        "|--------|-----|-------|-------|",
    ])
    for sname in systems:
        vals = []
        for beta in betas:
            key = f"{sname}__IC1__beta{beta}"
            vals.append(f"{all_data[key]['entropy'].mean():.2f}")
        lines.append(f"| {LABELS[sname]} | {' | '.join(vals)} |")

    lines.extend([
        "\n## Breather Counts (IC1, IPR > 0.3)\n",
        "| System | β=0 | β=0.5 | β=1.0 |",
        "|--------|-----|-------|-------|",
    ])
    for sname in systems:
        vals = []
        for beta in betas:
            key = f"{sname}__IC1__beta{beta}"
            vals.append(str(all_data[key]["breather_count"]))
        lines.append(f"| {LABELS[sname]} | {' | '.join(vals)} |")

    # Determine outcome
    ipr_600 = np.mean([all_data[f"600cell__IC1__beta{b}"]["ipr"].mean()
                        for b in betas])
    ipr_iso = np.mean([all_data[f"iso_full__IC1__beta{b}"]["ipr"].mean()
                        for b in betas])
    ipr_blk = np.mean([all_data[f"iso_block__IC1__beta{b}"]["ipr"].mean()
                        for b in betas])

    lines.append("\n## Outcome Assessment\n")
    lines.append(f"- 600-cell mean IPR (avg over β): {ipr_600:.4f}")
    lines.append(f"- Full surrogate mean IPR: {ipr_iso:.4f}")
    lines.append(f"- Block surrogate mean IPR: {ipr_blk:.4f}")

    ratio_full = ipr_600 / (ipr_iso + 1e-30)
    ratio_block = ipr_600 / (ipr_blk + 1e-30)
    lines.append(f"- 600-cell / full surrogate ratio: {ratio_full:.2f}×")
    lines.append(f"- 600-cell / block surrogate ratio: {ratio_block:.2f}×")

    if ratio_full > 2.0 and ratio_block > 2.0:
        lines.append("\n**Result: Case A — Geometry is the primary cause.**")
        lines.append("Localization depends on specific H₄ eigenvector geometry.")
    elif ratio_full < 1.5 and ratio_block < 1.5:
        lines.append("\n**Result: Case B — Spectrum alone explains the effect.**")
        lines.append("Spectral compression is the primary driver.")
    else:
        lines.append("\n**Result: Case C — Mixed: spectrum + geometry.**")
        lines.append("Spectral compression contributes, but H₄ geometry "
                     "enhances the effect.")

    with open(os.path.join(RESULTS_DIR, "isospectral_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  Saved isospectral_summary.md")

    print("\nDone!")


if __name__ == "__main__":
    run()
