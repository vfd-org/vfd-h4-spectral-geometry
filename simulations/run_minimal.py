"""
Minimal experiment: just enough to produce all 4 figures.
T=50, dt=0.01 → 5000 steps per run, 8 runs total.
~15 minutes on slow hardware.
"""

import os
import json
import time
import numpy as np
from scipy import sparse

from build_600cell import build_600cell
from build_control_graphs import rewired_graph
from laplacian import build_laplacian, compute_spectrum, spectral_summary

RESULTS_DIR = "results"


def integrate_sparse(Phi0, Pi0, L_sp, omega0, lam, beta, dt, n_steps,
                     save_interval=50):
    """Fast integrator using scipy sparse Laplacian."""
    Phi = Phi0.copy()
    Pi = Pi0.copy()
    n = len(Phi)

    n_saves = n_steps // save_interval + 1
    times = np.zeros(n_saves)
    Phi_hist = np.zeros((n_saves, n))
    energy = np.zeros(n_saves)

    def H():
        k = 0.5 * np.dot(Pi, Pi)
        p = 0.5 * omega0**2 * np.dot(Phi, Phi)
        c = 0.5 * lam * Phi.dot(L_sp.dot(Phi))
        nl = 0.25 * beta * np.sum(Phi**4)
        return k + p + c + nl

    Phi_hist[0] = Phi
    energy[0] = H()
    si = 1

    w2 = omega0**2
    for step in range(1, n_steps + 1):
        F = -w2 * Phi - lam * L_sp.dot(Phi) - beta * Phi * Phi * Phi
        Pi += 0.5 * dt * F
        Phi += dt * Pi
        F = -w2 * Phi - lam * L_sp.dot(Phi) - beta * Phi * Phi * Phi
        Pi += 0.5 * dt * F

        if step % save_interval == 0 and si < n_saves:
            times[si] = step * dt
            Phi_hist[si] = Phi
            energy[si] = H()
            si += 1

    return {
        "times": times[:si],
        "Phi_history": Phi_hist[:si],
        "energy": energy[:si],
    }


def ipr_series(Phi_hist):
    s2 = np.sum(Phi_hist**2, axis=1)
    s4 = np.sum(Phi_hist**4, axis=1)
    mask = s2 > 1e-30
    out = np.zeros(len(s2))
    out[mask] = s4[mask] / s2[mask]**2
    return out


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Building 600-cell...")
    A_600, vertices = build_600cell()

    print("Building rewired control...")
    A_ctrl = rewired_graph(A_600, num_swaps=10000, seed=99)

    graphs = {"600cell": A_600, "control1": A_ctrl}

    # Spectra
    spectra = {}
    L_sparse = {}
    for name, A in graphs.items():
        L = build_laplacian(A)
        evals, evecs = compute_spectrum(L)
        spectra[name] = (evals, evecs)
        L_sparse[name] = sparse.csr_matrix(L)
        distinct = spectral_summary(evals)
        print(f"  {name}: {len(distinct)} distinct eigenvalues")

    # Integration params
    omega0, lam, dt = 1.0, 1.0, 0.01
    T = 50  # short but enough to see dynamics
    n_steps = int(T / dt)
    save_interval = 10

    betas = [0.0, 0.5]
    all_data = {}

    for beta in betas:
        for gname in graphs:
            Lsp = L_sparse[gname]
            A = graphs[gname]

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
            print(f"{elapsed:.1f}s  drift={drift:.2e}  IPR_end={ipr[-1]:.4f}")

            # IC4: local neighbourhood
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
            print(f"{elapsed:.1f}s  drift={drift:.2e}  IPR_end={ipr[-1]:.4f}")

    # ── Plots ──────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure A: IPR comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("Inverse Participation Ratio: 600-cell vs Rewired Control",
                 fontsize=13)
    for ax, beta in zip(axes, betas):
        for gname, color, label in [
            ("600cell", "#7b2d8e", "600-cell ($H_4$)"),
            ("control1", "#999999", "Rewired control"),
        ]:
            for ic, ls in [("IC1", "-"), ("IC4", "--")]:
                key = f"{gname}__{ic}__beta{beta}"
                d = all_data[key]
                t = d["res"]["times"]
                ax.plot(t, d["ipr"], color=color, linestyle=ls,
                        label=f"{label} {ic}", alpha=0.8, linewidth=1)
        ax.axhline(1/120, color="gray", ls=":", lw=0.5)
        ax.set_title(f"$\\beta={beta}$")
        ax.set_xlabel("Time")
        ax.set_ylabel("IPR")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "localization_comparison.png"),
                dpi=200)
    plt.close(fig)
    print("Saved localization_comparison.png")

    # Figure B: Spectral occupancy heatmap
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Spectral Occupancy $|c_k|^2$ (IC1, $\\beta=0.5$)",
                 fontsize=13)
    for ax, gname, title in zip(axes, ["600cell", "control1"],
                                ["600-cell ($H_4$)", "Rewired control"]):
        key = f"{gname}__IC1__beta0.5"
        Phi_h = all_data[key]["res"]["Phi_history"]
        evecs = spectra[gname][1]
        evals = spectra[gname][0]
        order = np.argsort(evals)
        coeffs = Phi_h @ evecs
        power = coeffs[:, order]**2
        from matplotlib.colors import LogNorm
        vmin = max(power[power > 0].min(), 1e-12)
        t = all_data[key]["res"]["times"]
        im = ax.imshow(power.T, aspect="auto", origin="lower",
                       extent=[t[0], t[-1], 0, 120],
                       norm=LogNorm(vmin=vmin, vmax=power.max()),
                       cmap="viridis")
        ax.set_xlabel("Time")
        ax.set_ylabel("Eigenmode index")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="$|c_k|^2$", shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "spectral_occupancy.png"), dpi=200)
    plt.close(fig)
    print("Saved spectral_occupancy.png")

    # Figure C: 3D localization snapshot
    key = "600cell__IC1__beta0.5"
    Phi_h = all_data[key]["res"]["Phi_history"]
    t_idx = int(0.8 * len(Phi_h))
    Phi = Phi_h[t_idx]
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    amp = np.abs(Phi)
    amp_n = amp / (amp.max() + 1e-30)

    fig = plt.figure(figsize=(8, 7))
    ax3 = fig.add_subplot(111, projection="3d")
    sc = ax3.scatter(x, y, z, c=amp_n, cmap="magma",
                     s=10 + 200 * amp_n**2, alpha=0.7, edgecolors="none")
    ax3.set_xlabel("$x_1$"); ax3.set_ylabel("$x_2$"); ax3.set_zlabel("$x_3$")
    t_val = all_data[key]["res"]["times"][t_idx]
    ax3.set_title(f"Field on 600-Cell (3D proj.), t={t_val:.0f}, $\\beta=0.5$")
    fig.colorbar(sc, ax=ax3, label="$|\\Phi_i|$", shrink=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "example_localization.png"), dpi=200)
    plt.close(fig)
    print("Saved example_localization.png")

    # Figure D: Energy conservation
    fig, ax = plt.subplots(figsize=(10, 4))
    for gname, color in [("600cell", "#7b2d8e"), ("control1", "#999999")]:
        for beta, ls in [(0.0, "-"), (0.5, "--")]:
            key = f"{gname}__IC1__beta{beta}"
            e = all_data[key]["res"]["energy"]
            t = all_data[key]["res"]["times"]
            rel = (e - e[0]) / (abs(e[0]) + 1e-30)
            label = f"{'600-cell' if '600' in gname else 'Control'} β={beta}"
            ax.plot(t, rel, color=color, ls=ls, label=label, lw=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Relative energy drift $(H-H_0)/|H_0|$")
    ax.set_title("Energy Conservation Check")
    ax.legend(fontsize=8)
    ax.axhline(0, color="gray", lw=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "energy_conservation.png"), dpi=200)
    plt.close(fig)
    print("Saved energy_conservation.png")

    # Summary JSON
    summary = {
        "parameters": {"omega0": omega0, "lambda": lam, "dt": dt, "T": T},
        "600cell_spectrum": [
            {"eigenvalue": float(v), "multiplicity": int(m)}
            for v, m in spectral_summary(spectra["600cell"][0])
        ],
        "runs": {}
    }
    for key, d in all_data.items():
        e = d["res"]["energy"]
        summary["runs"][key] = {
            "ipr_final": float(d["ipr"][-1]),
            "ipr_mean": float(d["ipr"].mean()),
            "energy_drift": float(abs(e[-1]-e[0])/(abs(e[0])+1e-30)),
        }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved summary.json")

    print("\nDone!")


if __name__ == "__main__":
    run()
