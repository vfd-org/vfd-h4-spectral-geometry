"""
Control graph baseline comparison for Paper III.

Runs identical dynamics on the H4 (600-cell) graph and M random
12-regular control graphs, classifies attractors with the same
criteria, and produces comparative figures.

GPU-accelerated using the batched integrator.

Usage:
    python paper3_baseline_comparison.py [--n_controls 10] [--n_ic 40]
                                          [--n_beta 10] [--T 500]
"""

import argparse
import os
import time
import numpy as np
import torch

from build_600cell import build_600cell
from build_control_graphs import random_regular_graph
from laplacian import build_laplacian, compute_spectrum
from diagnostics import (
    inverse_participation_ratio,
    spectral_sector_energy,
    energy_drift,
)

# ── Parameters ────────────────────────────────────────────
OMEGA0 = 1.0
LAMBDA = 1.0
DT = 0.005
SAVE_INTERVAL = 200
T_TRANSIENT = 200
ENERGY_DRIFT_THRESHOLD = 5e-4

# Classification thresholds (identical to Paper III production)
IPR_DELOCAL = 0.03
IPR_MODERATE = 0.10
IPR_STRONG = 0.05


def integrate_batch_gpu(Phi_batch, Pi_batch, L_gpu, omega0, lam, beta,
                        dt, n_steps, save_interval, device):
    """Velocity-Verlet batch integration on GPU."""
    w2 = omega0 ** 2
    half_dt = 0.5 * dt
    batch_size, n = Phi_batch.shape

    n_saves = n_steps // save_interval + 1
    times = np.zeros(n_saves)
    Phi_hist = np.zeros((n_saves, batch_size, n))
    energy_hist = np.zeros((n_saves, batch_size))

    Phi = Phi_batch.clone()
    Pi = Pi_batch.clone()

    def compute_H():
        ke = 0.5 * torch.sum(Pi ** 2, dim=1)
        pe_base = 0.5 * w2 * torch.sum(Phi ** 2, dim=1)
        pe_coup = 0.5 * lam * torch.sum(Phi * (Phi @ L_gpu.T), dim=1)
        pe_nl = 0.25 * beta * torch.sum(Phi ** 4, dim=1)
        return (ke + pe_base + pe_coup + pe_nl).cpu().numpy()

    Phi_hist[0] = Phi.cpu().numpy()
    energy_hist[0] = compute_H()
    si = 1

    F = -w2 * Phi - lam * (Phi @ L_gpu.T) - beta * Phi * Phi * Phi

    for step in range(1, n_steps + 1):
        Pi.add_(F, alpha=half_dt)
        Phi.add_(Pi, alpha=dt)
        F = -w2 * Phi - lam * (Phi @ L_gpu.T) - beta * Phi * Phi * Phi
        Pi.add_(F, alpha=half_dt)

        if step % save_interval == 0 and si < n_saves:
            times[si] = step * dt
            Phi_hist[si] = Phi.cpu().numpy()
            energy_hist[si] = compute_H()
            si += 1

    return {
        "times": times[:si],
        "Phi_history": Phi_hist[:si],
        "energy": energy_hist[:si],
    }


def generate_ic_batch(n_vertices, eigenvectors, A, batch_size, rng):
    """Generate diverse initial conditions."""
    Phi_batch = np.zeros((batch_size, n_vertices))
    Pi_batch = np.zeros((batch_size, n_vertices))
    ic_types = ['IC1', 'IC2', 'IC3', 'IC4',
                'IC5', 'IC6', 'IC7', 'IC8']

    for b in range(batch_size):
        ic_type = ic_types[b % len(ic_types)]
        amplitude = rng.uniform(0.1, 2.0)

        if ic_type == 'IC1':
            j = rng.integers(0, n_vertices)
            Phi_batch[b, j] = amplitude
        elif ic_type == 'IC2':
            Phi_batch[b] = rng.normal(0, 0.1 * amplitude, n_vertices)
        elif ic_type == 'IC3':
            n_low = min(14, eigenvectors.shape[1])
            coeffs = rng.normal(0, amplitude, n_low)
            Phi_batch[b] = eigenvectors[:, :n_low] @ coeffs
        elif ic_type == 'IC4':
            j = rng.integers(0, n_vertices)
            Phi_batch[b, j] = amplitude
            neighbours = np.where(A[j] > 0)[0]
            Phi_batch[b, neighbours] = amplitude * 0.5
        elif ic_type == 'IC5':
            n_start = min(14, eigenvectors.shape[1])
            n_end = min(91, eigenvectors.shape[1])
            n_modes = n_end - n_start
            if n_modes > 0:
                coeffs = rng.normal(0, amplitude * 0.5, n_modes)
                Phi_batch[b] = eigenvectors[:, n_start:n_end] @ coeffs
        elif ic_type == 'IC6':
            k = rng.integers(0, min(91, eigenvectors.shape[1]))
            Phi_batch[b] = amplitude * eigenvectors[:, k]
            Phi_batch[b] += 0.05 * amplitude * rng.normal(size=n_vertices)
        elif ic_type == 'IC7':
            j = rng.integers(0, n_vertices)
            neighbours = np.where(A[j] > 0)[0]
            chosen = rng.choice(neighbours,
                               min(4, len(neighbours)), replace=False)
            Phi_batch[b, j] = amplitude
            Phi_batch[b, chosen] = amplitude * rng.uniform(0.3, 0.8,
                                                            len(chosen))
        elif ic_type == 'IC8':
            Phi_batch[b] = rng.normal(0, 0.2 * amplitude, n_vertices)
            Pi_batch[b] = rng.normal(0, 0.1 * amplitude, n_vertices)

    return Phi_batch, Pi_batch


def classify(ipr_mean, n_dominant, backbone_frac, trap_frac, max_autocorr):
    """Classify trajectory — identical criteria for all graphs."""
    if ipr_mean > IPR_STRONG and trap_frac > 0.35:
        return 3
    if ipr_mean > 0.05 and trap_frac > 0.4 and max_autocorr > 0.3:
        return 3
    if ipr_mean < IPR_DELOCAL and n_dominant <= 2 and backbone_frac > 0.7:
        if max_autocorr > 0.2:
            return 1
    if (ipr_mean < IPR_MODERATE and 2 <= n_dominant <= 6
            and backbone_frac > 0.5):
        return 2
    if ipr_mean < IPR_DELOCAL and n_dominant >= 2 and backbone_frac > 0.6:
        return 2
    if ipr_mean < IPR_DELOCAL and backbone_frac > 0.7:
        return 1
    return 4


def run_graph(graph_name, A, L, eigenvalues, eigenvectors, beta_values,
              n_ic, batch_size, T, device, rng):
    """Run full sweep for one graph. Returns class fractions and IPR arrays."""
    n_vertices = A.shape[0]
    L_gpu = torch.tensor(L, dtype=torch.float64, device=device)
    n_steps = int(T / DT)
    n_transient_saves = int(T_TRANSIENT / (DT * SAVE_INTERVAL))
    n_beta = len(beta_values)

    class_fractions = np.zeros((n_beta, 4))
    all_ipr = np.zeros((n_beta, n_ic))

    for b_idx, beta in enumerate(beta_values):
        for batch_start in range(0, n_ic, batch_size):
            actual_batch = min(batch_size, n_ic - batch_start)

            Phi_np, Pi_np = generate_ic_batch(
                n_vertices, eigenvectors, A, actual_batch, rng
            )
            Phi_gpu = torch.tensor(Phi_np, dtype=torch.float64, device=device)
            Pi_gpu = torch.tensor(Pi_np, dtype=torch.float64, device=device)

            res = integrate_batch_gpu(
                Phi_gpu, Pi_gpu, L_gpu, OMEGA0, LAMBDA, float(beta),
                DT, n_steps, SAVE_INTERVAL, device
            )

            # Diagnostics per trajectory
            for i in range(actual_batch):
                ic_idx = batch_start + i
                Phi_post = res['Phi_history'][n_transient_saves:, i, :]
                energy_post = res['energy'][n_transient_saves:, i]

                if len(Phi_post) < 10:
                    all_ipr[b_idx, ic_idx] = 0.0
                    continue

                drift = energy_drift(energy_post)
                if drift > ENERGY_DRIFT_THRESHOLD:
                    continue

                ipr = inverse_participation_ratio(Phi_post)
                ipr_mean = float(np.mean(ipr))
                all_ipr[b_idx, ic_idx] = ipr_mean

                # Spectral composition
                coeffs = Phi_post @ eigenvectors
                sector_E, labels = spectral_sector_energy(
                    coeffs, eigenvalues
                )
                sector_mean = np.mean(sector_E, axis=0)
                total = np.sum(sector_mean)
                if total > 1e-30:
                    fracs = sector_mean / total
                else:
                    fracs = np.zeros(len(labels))

                n_dominant = int(np.sum(fracs > 0.1))
                n_backbone = min(6, len(fracs))
                backbone_frac = float(np.sum(fracs[:n_backbone]))

                # Autocorrelation
                norms = np.linalg.norm(Phi_post, axis=1)
                n_post = len(Phi_post)
                max_lag = min(n_post // 4, 50)
                max_autocorr = 0.0
                for lag in range(2, max_lag, max(1, max_lag // 25)):
                    dots = np.sum(Phi_post[:-lag] * Phi_post[lag:], axis=1)
                    norm_prod = norms[:-lag] * norms[lag:]
                    mask = norm_prod > 1e-20
                    if np.sum(mask) > 0:
                        ac = float(np.mean(dots[mask] / norm_prod[mask]))
                        max_autocorr = max(max_autocorr, ac)

                # Energy trapping
                energy_proxy = np.mean(Phi_post ** 2, axis=0)
                total_ep = np.sum(energy_proxy)
                if total_ep > 1e-30:
                    top_v = np.argmax(energy_proxy)
                    A2 = A @ A
                    nbhd = ((A[top_v] + A2[top_v]) > 0).astype(float)
                    nbhd[top_v] = 1.0
                    trap_frac = float(np.sum(energy_proxy * nbhd) / total_ep)
                else:
                    trap_frac = 0.0

                cls = classify(ipr_mean, n_dominant, backbone_frac,
                               trap_frac, max_autocorr)
                # Store class (0-indexed in fractions)
                class_fractions[b_idx, cls - 1] += 1

        # Normalise
        row_total = np.sum(class_fractions[b_idx])
        if row_total > 0:
            class_fractions[b_idx] /= row_total

    return class_fractions, all_ipr


def run_comparison(n_controls=10, n_ic=40, n_beta=10, T=500,
                   batch_size=10, output_dir='results/paper3_baseline'):
    """Run the full H4 vs control graph comparison."""
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}"
          f"{' — ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")

    # Beta values — span all regimes
    beta_values = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0,
                            2.0, 3.5, 5.0])[:n_beta]

    # ── H4 graph ──────────────────────────────────────────
    print("\n=== H4 (600-cell) graph ===")
    A_h4, _ = build_600cell()
    L_h4 = build_laplacian(A_h4).astype(np.float64)
    evals_h4, evecs_h4 = compute_spectrum(L_h4)
    n_distinct_h4 = len(set(round(float(e), 6) for e in evals_h4))
    print(f"  Distinct eigenvalues: {n_distinct_h4}")

    rng_h4 = np.random.default_rng(42)
    t0 = time.time()
    h4_fracs, h4_ipr = run_graph(
        "H4", A_h4, L_h4, evals_h4, evecs_h4,
        beta_values, n_ic, batch_size, T, device, rng_h4
    )
    h4_time = time.time() - t0
    print(f"  Completed in {h4_time:.1f}s")
    for b_idx, beta in enumerate(beta_values):
        print(f"  beta={beta:.3f}: C1={h4_fracs[b_idx,0]:.2f} "
              f"C2={h4_fracs[b_idx,1]:.2f} C3={h4_fracs[b_idx,2]:.2f} "
              f"C4={h4_fracs[b_idx,3]:.2f}")

    # ── Control graphs ────────────────────────────────────
    print(f"\n=== Random 12-regular controls (M={n_controls}) ===")
    ctrl_fracs_all = np.zeros((n_controls, n_beta, 4))
    ctrl_ipr_all = np.zeros((n_controls, n_beta, n_ic))
    ctrl_n_distinct = np.zeros(n_controls)

    for m in range(n_controls):
        print(f"\n  Control {m+1}/{n_controls}")
        seed = 1000 + m
        A_ctrl = random_regular_graph(120, 12, seed=seed)
        L_ctrl = build_laplacian(A_ctrl).astype(np.float64)
        evals_ctrl, evecs_ctrl = compute_spectrum(L_ctrl)
        n_dist = len(set(round(float(e), 6) for e in evals_ctrl))
        ctrl_n_distinct[m] = n_dist
        print(f"    Distinct eigenvalues: {n_dist}")

        rng_ctrl = np.random.default_rng(42)  # same IC seed for fairness
        t0 = time.time()
        ctrl_fracs, ctrl_ipr = run_graph(
            f"RRG_{m}", A_ctrl, L_ctrl, evals_ctrl, evecs_ctrl,
            beta_values, n_ic, batch_size, T, device, rng_ctrl
        )
        elapsed = time.time() - t0
        ctrl_fracs_all[m] = ctrl_fracs
        ctrl_ipr_all[m] = ctrl_ipr
        print(f"    Completed in {elapsed:.1f}s")

    # ── Aggregate statistics ──────────────────────────────
    ctrl_fracs_mean = np.mean(ctrl_fracs_all, axis=0)
    ctrl_fracs_std = np.std(ctrl_fracs_all, axis=0)
    ctrl_ipr_flat = ctrl_ipr_all.reshape(n_controls * n_beta, n_ic)

    # Attractor diversity (entropy of class distribution)
    def entropy(fracs):
        p = fracs[fracs > 0]
        return -np.sum(p * np.log2(p))

    h4_entropy = np.array([entropy(h4_fracs[b]) for b in range(n_beta)])
    ctrl_entropy_mean = np.array([
        np.mean([entropy(ctrl_fracs_all[m, b])
                 for m in range(n_controls)])
        for b in range(n_beta)
    ])

    # ── Save results ──────────────────────────────────────
    np.savez(os.path.join(output_dir, 'comparison_results.npz'),
             beta_values=beta_values,
             h4_fracs=h4_fracs,
             h4_ipr=h4_ipr,
             ctrl_fracs_mean=ctrl_fracs_mean,
             ctrl_fracs_std=ctrl_fracs_std,
             ctrl_fracs_all=ctrl_fracs_all,
             ctrl_ipr_all=ctrl_ipr_all,
             ctrl_n_distinct=ctrl_n_distinct,
             h4_entropy=h4_entropy,
             ctrl_entropy_mean=ctrl_entropy_mean,
             h4_n_distinct=n_distinct_h4)

    print(f"\nResults saved to {output_dir}/")

    # ── Print summary ─────────────────────────────────────
    print("\n=== SUMMARY ===")
    print(f"H4 distinct eigenvalues: {n_distinct_h4}")
    print(f"RRG distinct eigenvalues: {np.mean(ctrl_n_distinct):.1f} "
          f"± {np.std(ctrl_n_distinct):.1f}")
    print(f"\nMean class fractions (averaged over beta):")
    print(f"  H4:  C1={np.mean(h4_fracs[:,0]):.3f} "
          f"C2={np.mean(h4_fracs[:,1]):.3f} "
          f"C3={np.mean(h4_fracs[:,2]):.3f} "
          f"C4={np.mean(h4_fracs[:,3]):.3f}")
    print(f"  RRG: C1={np.mean(ctrl_fracs_mean[:,0]):.3f} "
          f"C2={np.mean(ctrl_fracs_mean[:,1]):.3f} "
          f"C3={np.mean(ctrl_fracs_mean[:,2]):.3f} "
          f"C4={np.mean(ctrl_fracs_mean[:,3]):.3f}")
    print(f"\nMean IPR (all beta):")
    print(f"  H4:  {np.mean(h4_ipr[h4_ipr>0]):.4f}")
    ctrl_ipr_nonzero = ctrl_ipr_all[ctrl_ipr_all > 0]
    print(f"  RRG: {np.mean(ctrl_ipr_nonzero):.4f}")

    return beta_values, h4_fracs, ctrl_fracs_mean, ctrl_fracs_std


def generate_figures(input_dir='results/paper3_baseline',
                     output_dir='../papers/paper-003-h4-attractors/figures'):
    """Generate comparison figures from saved results."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
    })

    os.makedirs(output_dir, exist_ok=True)
    data = np.load(os.path.join(input_dir, 'comparison_results.npz'))

    beta = data['beta_values']
    h4_fracs = data['h4_fracs']
    ctrl_mean = data['ctrl_fracs_mean']
    ctrl_std = data['ctrl_fracs_std']
    h4_ipr = data['h4_ipr']
    ctrl_ipr = data['ctrl_ipr_all']

    CLASS_LABELS = ['Class 1: Backbone', 'Class 2: Locked',
                    'Class 3: Breather', 'Class 4: Transitional']
    H4_COLORS = ['#2166ac', '#4dac26', '#d6604d', '#999999']
    CTRL_COLORS = ['#92c5de', '#a6d96a', '#f4a582', '#cccccc']

    # ── Figure A: Class Distribution Comparison ───────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle('Attractor Class Distribution: H$_4$ vs Random Regular Graphs',
                 fontsize=14)

    for c, ax in enumerate(axes.flat):
        ax.plot(beta, h4_fracs[:, c], 'o-', color=H4_COLORS[c],
                linewidth=2, markersize=5, label='H$_4$ (600-cell)')
        ax.plot(beta, ctrl_mean[:, c], 's--', color=CTRL_COLORS[c],
                linewidth=1.5, markersize=4, label='RRG (mean)')
        ax.fill_between(beta,
                         ctrl_mean[:, c] - ctrl_std[:, c],
                         ctrl_mean[:, c] + ctrl_std[:, c],
                         color=CTRL_COLORS[c], alpha=0.3,
                         label='RRG (±1σ)')
        ax.set_xscale('log')
        ax.set_xlabel(r'$\beta$')
        ax.set_ylabel('Fraction')
        ax.set_title(CLASS_LABELS[c])
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, 'fig_baseline_classes.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Figure B: IPR Distribution Comparison ─────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle('IPR Distribution: H$_4$ vs Random Regular Graphs',
                 fontsize=14)

    # Pick 3 beta values: low, mid, high
    targets = [0, len(beta) // 2, len(beta) - 1]
    bins = np.logspace(np.log10(0.005), np.log10(1.0), 30)

    for ax, b_idx in zip(axes, targets):
        h4_vals = h4_ipr[b_idx][h4_ipr[b_idx] > 0]
        # Pool all control IPR at this beta
        ctrl_vals = ctrl_ipr[:, b_idx, :].flatten()
        ctrl_vals = ctrl_vals[ctrl_vals > 0]

        if len(h4_vals) > 0:
            ax.hist(h4_vals, bins=bins, alpha=0.6, color='#d6604d',
                    label='H$_4$', edgecolor='#d6604d', linewidth=0.5)
        if len(ctrl_vals) > 0:
            ax.hist(ctrl_vals, bins=bins, alpha=0.4, color='#92c5de',
                    label='RRG', edgecolor='#2166ac', linewidth=0.5)

        ax.axvline(x=1/120, color='k', linestyle=':', alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlabel('Time-averaged IPR')
        ax.set_ylabel('Count')
        ax.set_title(rf'$\beta = {beta[b_idx]:.3f}$')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    path = os.path.join(output_dir, 'fig_baseline_ipr.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Figure C: Spectral comparison ─────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    h4_n = data['h4_n_distinct']
    ctrl_n = data['ctrl_n_distinct']

    ax.bar([0], [h4_n], width=0.4, color='#d6604d', label='H$_4$')
    ax.bar(np.arange(1, len(ctrl_n) + 1), ctrl_n, width=0.4,
           color='#92c5de', label='RRG')
    ax.axhline(y=h4_n, color='#d6604d', linestyle='--', alpha=0.5)
    ax.set_xlabel('Graph index (0 = H$_4$)')
    ax.set_ylabel('Distinct eigenvalues')
    ax.set_title('Spectral Compression: H$_4$ vs Random Regular Graphs')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    path = os.path.join(output_dir, 'fig_baseline_spectrum.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Paper III baseline comparison'
    )
    parser.add_argument('--n_controls', type=int, default=10)
    parser.add_argument('--n_ic', type=int, default=40)
    parser.add_argument('--n_beta', type=int, default=10)
    parser.add_argument('--T', type=float, default=500)
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--output', type=str,
                        default='results/paper3_baseline')
    parser.add_argument('--figures', type=str,
                        default='../papers/paper-003-h4-attractors/figures')
    args = parser.parse_args()

    beta_vals, h4_f, ctrl_m, ctrl_s = run_comparison(
        n_controls=args.n_controls,
        n_ic=args.n_ic,
        n_beta=args.n_beta,
        T=args.T,
        batch_size=args.batch,
        output_dir=args.output,
    )

    print("\n=== Generating figures ===")
    generate_figures(input_dir=args.output, output_dir=args.figures)
    print("\nDone.")
