"""
WO-VFD-SIM-002: Long-time stability, shell propagation, symmetry verification.

Runs: 600-cell + rewired control, IC1+IC4, β∈{0, 0.5, 1.0}, T=500.
Produces all diagnostic figures and breather statistics.
"""

import os
import json
import time
import numpy as np
from scipy import sparse

from build_600cell import build_600cell
from build_control_graphs import rewired_graph, random_regular_graph
from laplacian import build_laplacian, compute_spectrum, spectral_summary
from shell_analysis import compute_graph_distances, shell_partition, shell_energy
from symmetry_tests import (find_equivalent_vertex, compare_trajectories,
                            detect_breathers)

RESULTS_DIR = "results_long"

# ── Integration ──────────────────────────────────────────────

def integrate_sparse(Phi0, Pi0, L_sp, omega0, lam, beta, dt, n_steps,
                     save_interval=100):
    """Velocity-Verlet with scipy sparse Laplacian."""
    Phi = Phi0.astype(np.float64).copy()
    Pi = Pi0.astype(np.float64).copy()
    n = len(Phi)
    w2 = omega0**2

    n_saves = n_steps // save_interval + 1
    times = np.zeros(n_saves)
    Phi_hist = np.zeros((n_saves, n))
    energy = np.zeros(n_saves)

    def H():
        return (0.5 * np.dot(Pi, Pi)
                + 0.5 * w2 * np.dot(Phi, Phi)
                + 0.5 * lam * Phi.dot(L_sp.dot(Phi))
                + 0.25 * beta * np.sum(Phi**4))

    Phi_hist[0] = Phi
    energy[0] = H()
    si = 1

    F = -w2 * Phi - lam * L_sp.dot(Phi) - beta * Phi * Phi * Phi
    for step in range(1, n_steps + 1):
        Pi += 0.5 * dt * F
        Phi += dt * Pi
        F = -w2 * Phi - lam * L_sp.dot(Phi) - beta * Phi * Phi * Phi
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


def spectral_entropy(coeffs, eigenvalues, tol=1e-6):
    """Compute Shannon entropy of spectral sector occupation."""
    sorted_idx = np.argsort(eigenvalues)
    sorted_ev = eigenvalues[sorted_idx]

    sectors = []
    i = 0
    while i < len(sorted_ev):
        j = i + 1
        while j < len(sorted_ev) and abs(sorted_ev[j] - sorted_ev[i]) < tol:
            j += 1
        sectors.append(sorted_idx[i:j])
        i = j

    n_times = coeffs.shape[0]
    entropy = np.zeros(n_times)
    sector_E = np.zeros((n_times, len(sectors)))

    for s_idx, indices in enumerate(sectors):
        sector_E[:, s_idx] = np.sum(coeffs[:, indices]**2, axis=1)

    for t in range(n_times):
        total = sector_E[t].sum()
        if total < 1e-30:
            continue
        p = sector_E[t] / total
        p = p[p > 1e-30]
        entropy[t] = -np.sum(p * np.log(p))

    return entropy, sector_E


# ── Main ─────────────────────────────────────────────────────

def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("WO-VFD-SIM-002: Long-time stability experiment")
    print("=" * 60)

    # Build graphs
    print("\nBuilding graphs...")
    A_600, vertices = build_600cell()
    A_ctrl1 = rewired_graph(A_600, num_swaps=10000, seed=99)
    A_ctrl2 = random_regular_graph(120, 12, seed=42)

    graphs = {"600cell": A_600, "control1": A_ctrl1, "control2": A_ctrl2}

    # Spectra + sparse Laplacians
    spectra = {}
    L_sparse = {}
    for name, A in graphs.items():
        L = build_laplacian(A)
        evals, evecs = compute_spectrum(L)
        spectra[name] = (evals, evecs)
        L_sparse[name] = sparse.csr_matrix(L.astype(np.float64))
        distinct = spectral_summary(evals)
        print(f"  {name}: {len(distinct)} distinct eigenvalues")

    # Shell structure for 600-cell
    print("\nShell analysis (600-cell, vertex 0):")
    dist_600, shells_600 = compute_graph_distances(A_600, 0), None
    shells_600 = shell_partition(dist_600)
    for d in sorted(shells_600):
        print(f"  d={d}: {len(shells_600[d])} vertices")

    # Parameters
    omega0, lam, dt = 1.0, 1.0, 0.01
    T = 500
    n_steps = int(T / dt)
    save_interval = 50  # save every 0.5 time units
    betas = [0.0, 0.5, 1.0]

    all_data = {}

    # ── Core runs ────────────────────────────────────────────
    print(f"\nRunning simulations (T={T}, {n_steps} steps)...")

    for beta in betas:
        for gname in graphs:
            A = graphs[gname]
            Lsp = L_sparse[gname]

            # IC1: single vertex
            Phi0 = np.zeros(120); Phi0[0] = 1.0
            key = f"{gname}__IC1__beta{beta}"
            print(f"  {key}...", end=" ", flush=True)
            t0 = time.time()
            res = integrate_sparse(Phi0, np.zeros(120), Lsp,
                                   omega0, lam, beta, dt, n_steps,
                                   save_interval)
            elapsed = time.time() - t0
            ipr = ipr_series(res["Phi_history"])
            drift = abs(res["energy"][-1] - res["energy"][0]) / (abs(res["energy"][0]) + 1e-30)
            all_data[key] = {"res": res, "ipr": ipr}
            print(f"{elapsed:.1f}s  drift={drift:.2e}  "
                  f"IPR_mean={ipr.mean():.4f}")

            # IC4: local H3 neighbourhood
            Phi0 = np.zeros(120); Phi0[0] = 1.0
            Phi0[np.where(A[0] > 0)[0]] = 0.5
            key = f"{gname}__IC4__beta{beta}"
            print(f"  {key}...", end=" ", flush=True)
            t0 = time.time()
            res = integrate_sparse(Phi0, np.zeros(120), Lsp,
                                   omega0, lam, beta, dt, n_steps,
                                   save_interval)
            elapsed = time.time() - t0
            ipr = ipr_series(res["Phi_history"])
            drift = abs(res["energy"][-1] - res["energy"][0]) / (abs(res["energy"][0]) + 1e-30)
            all_data[key] = {"res": res, "ipr": ipr}
            print(f"{elapsed:.1f}s  drift={drift:.2e}  "
                  f"IPR_mean={ipr.mean():.4f}")

    # ── Symmetry equivalence run ─────────────────────────────
    print("\nSymmetry equivalence test...")
    v1 = find_equivalent_vertex(A_600, 0)
    print(f"  Comparing vertex 0 vs vertex {v1}")

    Phi0_v0 = np.zeros(120); Phi0_v0[0] = 1.0
    Phi0_v1 = np.zeros(120); Phi0_v1[v1] = 1.0

    for beta in [0.5]:
        res_v0 = integrate_sparse(Phi0_v0, np.zeros(120), L_sparse["600cell"],
                                  omega0, lam, beta, dt, n_steps, save_interval)
        res_v1 = integrate_sparse(Phi0_v1, np.zeros(120), L_sparse["600cell"],
                                  omega0, lam, beta, dt, n_steps, save_interval)
        ipr_v0 = ipr_series(res_v0["Phi_history"])
        ipr_v1 = ipr_series(res_v1["Phi_history"])
        sym_stats = compare_trajectories(ipr_v0, ipr_v1)
        all_data["sym_v0"] = {"res": res_v0, "ipr": ipr_v0}
        all_data["sym_v1"] = {"res": res_v1, "ipr": ipr_v1}
        all_data["sym_stats"] = sym_stats
        print(f"  mean IPR v0={sym_stats['mean_ipr_v0']:.4f}, "
              f"v1={sym_stats['mean_ipr_v1']:.4f}, "
              f"diff={sym_stats['mean_difference']:.4f}, "
              f"corr={sym_stats['correlation']:.4f}")

    # ── Plots ────────────────────────────────────────────────
    print("\nGenerating plots...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    COLORS = {"600cell": "#7b2d8e", "control1": "#999999", "control2": "#bbbbbb"}
    LABELS = {"600cell": "600-cell ($H_4$)", "control1": "Rewired ctrl",
              "control2": "Random 12-reg"}

    # ── Figure 1: IPR long run ───────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    fig.suptitle(f"IPR Over Time (IC1, T={T})", fontsize=13)
    for ax, beta in zip(axes, betas):
        for gname in graphs:
            key = f"{gname}__IC1__beta{beta}"
            d = all_data[key]
            t = d["res"]["times"]
            step = max(1, len(t) // 2000)
            ax.plot(t[::step], d["ipr"][::step],
                    color=COLORS[gname], label=LABELS[gname],
                    alpha=0.7, lw=0.7)
        ax.axhline(1/120, color="gray", ls=":", lw=0.5)
        ax.set_title(f"$\\beta={beta}$")
        ax.set_xlabel("Time"); ax.set_ylabel("IPR")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "IPR_long_run.png"), dpi=200)
    plt.close(fig)
    print("  Saved IPR_long_run.png")

    # ── Figure 2: Spectral entropy ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    fig.suptitle(f"Spectral Sector Entropy (IC1, T={T})", fontsize=13)
    for ax, beta in zip(axes, betas):
        for gname in graphs:
            key = f"{gname}__IC1__beta{beta}"
            Phi_h = all_data[key]["res"]["Phi_history"]
            evecs = spectra[gname][1]
            evals = spectra[gname][0]
            coeffs = Phi_h @ evecs
            ent, _ = spectral_entropy(coeffs, evals)
            t = all_data[key]["res"]["times"]
            step = max(1, len(t) // 2000)
            ax.plot(t[::step], ent[::step], color=COLORS[gname],
                    label=LABELS[gname], alpha=0.7, lw=0.8)
        ax.set_title(f"$\\beta={beta}$")
        ax.set_xlabel("Time"); ax.set_ylabel("Spectral entropy")
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "spectral_entropy.png"), dpi=200)
    plt.close(fig)
    print("  Saved spectral_entropy.png")

    # ── Figure 3: Spectral band evolution (600-cell, β=0.5) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Spectral Sector Energy (IC1, $\\beta=0.5$)", fontsize=13)
    for ax, gname, title in zip(axes, ["600cell", "control1"],
                                ["600-cell ($H_4$)", "Rewired control"]):
        key = f"{gname}__IC1__beta0.5"
        Phi_h = all_data[key]["res"]["Phi_history"]
        evecs = spectra[gname][1]
        evals = spectra[gname][0]
        coeffs = Phi_h @ evecs
        _, sector_E = spectral_entropy(coeffs, evals)
        t = all_data[key]["res"]["times"]
        n_sec = sector_E.shape[1]
        for s in range(n_sec):
            ax.plot(t, sector_E[:, s], lw=0.8, alpha=0.7,
                    label=f"Sector {s+1}")
        ax.set_xlabel("Time"); ax.set_ylabel("Sector energy")
        ax.set_title(title)
        if n_sec <= 12:
            ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "spectral_band_evolution.png"),
                dpi=200)
    plt.close(fig)
    print("  Saved spectral_band_evolution.png")

    # ── Figure 4: Shell energy propagation ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Distance-Shell Energy Propagation (IC1, $\\beta=0.5$)",
                 fontsize=13)

    for ax, gname, title in zip(axes, ["600cell", "control1"],
                                ["600-cell ($H_4$)", "Rewired control"]):
        key = f"{gname}__IC1__beta0.5"
        Phi_h = all_data[key]["res"]["Phi_history"]
        t = all_data[key]["res"]["times"]

        dists = compute_graph_distances(graphs[gname], 0)
        shells = shell_partition(dists)
        sE = shell_energy(Phi_h, shells)

        cmap = plt.cm.viridis
        n_shells = len(sE)
        for d in sorted(sE.keys()):
            c = cmap(d / max(n_shells - 1, 1))
            ax.plot(t, sE[d], color=c, lw=0.9, alpha=0.8,
                    label=f"d={d} ({len(shells[d])}v)")
        ax.set_xlabel("Time"); ax.set_ylabel("Shell energy $E_d(t)$")
        ax.set_title(title)
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "shell_energy_vs_time.png"),
                dpi=200)
    plt.close(fig)
    print("  Saved shell_energy_vs_time.png")

    # ── Figure 5: Symmetry equivalence ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"Symmetry Equivalence: vertex 0 vs vertex {v1} "
                 f"($\\beta=0.5$)", fontsize=13)

    ax = axes[0]
    t = all_data["sym_v0"]["res"]["times"]
    ax.plot(t, all_data["sym_v0"]["ipr"], color="#7b2d8e",
            label=f"Vertex 0", alpha=0.7, lw=0.7)
    ax.plot(t, all_data["sym_v1"]["ipr"], color="#e07a3a",
            label=f"Vertex {v1}", alpha=0.7, lw=0.7)
    ax.set_xlabel("Time"); ax.set_ylabel("IPR")
    ax.set_title("IPR(t) comparison"); ax.legend(fontsize=8)

    ax = axes[1]
    # Histogram of IPR values
    ax.hist(all_data["sym_v0"]["ipr"], bins=40, alpha=0.5,
            color="#7b2d8e", label=f"Vertex 0", density=True)
    ax.hist(all_data["sym_v1"]["ipr"], bins=40, alpha=0.5,
            color="#e07a3a", label=f"Vertex {v1}", density=True)
    ax.set_xlabel("IPR"); ax.set_ylabel("Density")
    ax.set_title("IPR distribution"); ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "symmetry_equivalence.png"),
                dpi=200)
    plt.close(fig)
    print("  Saved symmetry_equivalence.png")

    # ── Breather detection ───────────────────────────────────
    print("\nBreather detection...")
    breather_stats = {}
    for beta in betas:
        for gname in graphs:
            key = f"{gname}__IC1__beta{beta}"
            d = all_data[key]
            t = d["res"]["times"]
            breathers = detect_breathers(d["ipr"], t,
                                        threshold=0.3, min_duration_steps=3)
            breather_stats[key] = {
                "count": len(breathers),
                "events": breathers[:20],  # cap at 20
            }
            if breathers:
                durations = [b["duration"] for b in breathers]
                peaks = [b["peak_ipr"] for b in breathers]
                print(f"  {key}: {len(breathers)} breathers, "
                      f"mean_dur={np.mean(durations):.2f}, "
                      f"peak_IPR={np.max(peaks):.3f}")
            else:
                print(f"  {key}: no breathers detected")

    with open(os.path.join(RESULTS_DIR, "breather_statistics.json"), "w") as f:
        json.dump(breather_stats, f, indent=2)
    print("  Saved breather_statistics.json")

    # ── Summary report ───────────────────────────────────────
    summary_lines = [
        "# WO-VFD-SIM-002 Results Summary\n",
        f"**Parameters:** ω₀={omega0}, λ={lam}, dt={dt}, T={T}\n",
        f"**Graphs:** 600-cell (9 eigenvalues), rewired control "
        f"({len(spectral_summary(spectra['control1'][0]))} eigenvalues), "
        f"random 12-regular "
        f"({len(spectral_summary(spectra['control2'][0]))} eigenvalues)\n",
        "\n## Localization Persistence\n",
        "| Run | Mean IPR | Std IPR | Energy Drift |",
        "|-----|----------|---------|--------------|",
    ]
    for beta in betas:
        for gname in graphs:
            key = f"{gname}__IC1__beta{beta}"
            d = all_data[key]
            e = d["res"]["energy"]
            drift = abs(e[-1]-e[0])/(abs(e[0])+1e-30)
            summary_lines.append(
                f"| {key} | {d['ipr'].mean():.4f} | "
                f"{d['ipr'].std():.4f} | {drift:.2e} |"
            )

    summary_lines.extend([
        "\n## Symmetry Equivalence\n",
        f"- Vertex 0 mean IPR: {sym_stats['mean_ipr_v0']:.4f}",
        f"- Vertex {v1} mean IPR: {sym_stats['mean_ipr_v1']:.4f}",
        f"- Mean difference: {sym_stats['mean_difference']:.4f}",
        f"- Correlation: {sym_stats['correlation']:.4f}",
        f"- Equivalent: {sym_stats['equivalent']}",
        "\n## Shell Structure (600-cell)\n",
    ])
    for d in sorted(shells_600):
        summary_lines.append(f"- distance {d}: {len(shells_600[d])} vertices")

    summary_lines.extend([
        "\n## Breather Counts (IC1)\n",
        "| Run | Count |",
        "|-----|-------|",
    ])
    for beta in betas:
        for gname in graphs:
            key = f"{gname}__IC1__beta{beta}"
            summary_lines.append(
                f"| {key} | {breather_stats[key]['count']} |"
            )

    with open(os.path.join(RESULTS_DIR, "results_summary.md"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print("  Saved results_summary.md")

    print("\nDone!")


if __name__ == "__main__":
    run()
