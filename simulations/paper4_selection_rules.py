"""
Paper IV: Selection Rules and Constraints on Nonlinear Attractors.

Computes mode coupling matrices, spatial shell occupancy, stability
hierarchy, and attractor family clustering for the H4 graph and
control graphs.

GPU-accelerated. Saves full sector fraction data per trajectory.

Usage:
    python paper4_selection_rules.py [--n_beta 15] [--n_ic 80] [--T 2000]
"""

import argparse
import os
import time
import numpy as np
import torch

from build_600cell import build_600cell
from build_control_graphs import random_regular_graph, rewired_graph
from laplacian import build_laplacian, compute_spectrum
from diagnostics import (
    inverse_participation_ratio,
    spectral_sector_energy,
    energy_drift,
)
from shell_analysis import compute_graph_distances, shell_partition

# ── Parameters ────────────────────────────────────────────
OMEGA0 = 1.0
LAMBDA = 1.0
DT = 0.005
SAVE_INTERVAL = 200
T_TRANSIENT = 200
ENERGY_DRIFT_THRESHOLD = 5e-4
SECTOR_THRESHOLD = 0.10  # sector "active" if >10% of energy

# Classification (same as Paper 3)
IPR_DELOCAL = 0.03
IPR_MODERATE = 0.10
IPR_STRONG = 0.08


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

    return {"times": times[:si], "Phi_history": Phi_hist[:si],
            "energy": energy_hist[:si]}


def generate_ic_batch(n_vertices, eigenvectors, A, batch_size, rng):
    """Generate diverse initial conditions (8 types)."""
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
            if n_end > n_start:
                coeffs = rng.normal(0, amplitude * 0.5, n_end - n_start)
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
    """Classify trajectory (identical to Paper 3)."""
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


def run_sweep_p4(graph_name, A, L, eigenvalues, eigenvectors,
                 beta_values, n_ic, batch_size, T, device, rng):
    """
    Run sweep saving full sector fractions and snapshots per trajectory.

    Returns dict with all Paper 4 analysis data.
    """
    n_vertices = A.shape[0]
    L_gpu = torch.tensor(L, dtype=torch.float64, device=device)
    n_steps = int(T / DT)
    n_transient_saves = int(T_TRANSIENT / (DT * SAVE_INTERVAL))
    n_beta = len(beta_values)

    # Compute sector structure
    sector_E_dummy, sector_labels = spectral_sector_energy(
        np.zeros((1, n_vertices)) @ eigenvectors, eigenvalues
    )
    n_sectors = len(sector_labels)

    # Storage
    all_classes = np.zeros((n_beta, n_ic), dtype=int)
    all_ipr = np.zeros((n_beta, n_ic))
    all_sector_fracs = np.zeros((n_beta, n_ic, n_sectors))
    all_snapshots = np.zeros((n_beta, n_ic, n_vertices))
    all_persistence = np.zeros((n_beta, n_ic))

    for b_idx, beta in enumerate(beta_values):
        for batch_start in range(0, n_ic, batch_size):
            actual_batch = min(batch_size, n_ic - batch_start)

            Phi_np, Pi_np = generate_ic_batch(
                n_vertices, eigenvectors, A, actual_batch, rng)
            Phi_gpu = torch.tensor(Phi_np, dtype=torch.float64, device=device)
            Pi_gpu = torch.tensor(Pi_np, dtype=torch.float64, device=device)

            res = integrate_batch_gpu(
                Phi_gpu, Pi_gpu, L_gpu, OMEGA0, LAMBDA, float(beta),
                DT, n_steps, SAVE_INTERVAL, device)

            for i in range(actual_batch):
                ic_idx = batch_start + i
                Phi_post = res['Phi_history'][n_transient_saves:, i, :]
                energy_post = res['energy'][n_transient_saves:, i]

                if len(Phi_post) < 10:
                    all_classes[b_idx, ic_idx] = 4
                    continue

                drift = energy_drift(energy_post)
                if drift > ENERGY_DRIFT_THRESHOLD:
                    all_classes[b_idx, ic_idx] = 4
                    continue

                # IPR
                ipr = inverse_participation_ratio(Phi_post)
                ipr_mean = float(np.mean(ipr))
                all_ipr[b_idx, ic_idx] = ipr_mean

                # Sector fractions (the key new data)
                coeffs = Phi_post @ eigenvectors
                sector_E, _ = spectral_sector_energy(coeffs, eigenvalues)
                sector_mean = np.mean(sector_E, axis=0)
                total = np.sum(sector_mean)
                if total > 1e-30:
                    fracs = sector_mean / total
                else:
                    fracs = np.zeros(n_sectors)
                all_sector_fracs[b_idx, ic_idx] = fracs

                # Snapshot (late-time)
                all_snapshots[b_idx, ic_idx] = Phi_post[-1]

                # Classification metrics
                n_dominant = int(np.sum(fracs > SECTOR_THRESHOLD))
                n_backbone = min(6, n_sectors)
                backbone_frac = float(np.sum(fracs[:n_backbone]))

                # Autocorrelation
                norms = np.linalg.norm(Phi_post, axis=1)
                max_lag = min(len(Phi_post) // 4, 50)
                max_autocorr = 0.0
                for lag in range(2, max_lag, max(1, max_lag // 25)):
                    dots = np.sum(Phi_post[:-lag] * Phi_post[lag:], axis=1)
                    norm_prod = norms[:-lag] * norms[lag:]
                    mask = norm_prod > 1e-20
                    if np.sum(mask) > 0:
                        ac = float(np.mean(dots[mask] / norm_prod[mask]))
                        max_autocorr = max(max_autocorr, ac)
                all_persistence[b_idx, ic_idx] = max_autocorr

                # Energy trapping
                ep = np.mean(Phi_post ** 2, axis=0)
                tep = np.sum(ep)
                if tep > 1e-30:
                    top_v = np.argmax(ep)
                    A2 = A @ A
                    nbhd = ((A[top_v] + A2[top_v]) > 0).astype(float)
                    nbhd[top_v] = 1.0
                    trap_frac = float(np.sum(ep * nbhd) / tep)
                else:
                    trap_frac = 0.0

                cls = classify(ipr_mean, n_dominant, backbone_frac,
                               trap_frac, max_autocorr)
                all_classes[b_idx, ic_idx] = cls

        # Progress
        c_counts = [np.sum(all_classes[b_idx] == c) for c in range(1, 5)]
        print(f"  [{graph_name}] beta={beta:.4f}: "
              f"C1={c_counts[0]} C2={c_counts[1]} "
              f"C3={c_counts[2]} C4={c_counts[3]}")

    return {
        'classes': all_classes,
        'ipr': all_ipr,
        'sector_fracs': all_sector_fracs,
        'snapshots': all_snapshots,
        'persistence': all_persistence,
        'sector_labels': sector_labels,
    }


def build_coupling_matrix(sector_fracs, classes, threshold=0.10):
    """
    Build 9x9 mode coupling matrix from sector fractions.

    M_ij = frequency that sectors i and j are both active
    (above threshold) in stable attractors.
    """
    n_beta, n_ic, n_sectors = sector_fracs.shape

    # Flatten and filter stable attractors (Classes 1-3)
    stable_mask = (classes >= 1) & (classes <= 3)
    stable_fracs = sector_fracs[stable_mask]
    n_stable = len(stable_fracs)

    if n_stable == 0:
        return np.zeros((n_sectors, n_sectors)), 0

    # Active sectors per trajectory
    active = (stable_fracs > threshold)  # (n_stable, n_sectors)

    # Coupling matrix: co-occurrence frequency
    M = np.zeros((n_sectors, n_sectors))
    for t in range(n_stable):
        active_sectors = np.where(active[t])[0]
        for i in active_sectors:
            for j in active_sectors:
                M[i, j] += 1

    M /= n_stable
    return M, n_stable


def compute_shell_occupancy(snapshots, classes, A):
    """
    Compute shell energy distribution per attractor class.

    For each snapshot, find the energy centroid vertex and compute
    energy fraction per distance shell from that vertex.
    """
    n_beta, n_ic, n_vertices = snapshots.shape
    distances_cache = {}

    # Pre-compute distances from each vertex
    for v in range(n_vertices):
        distances_cache[v] = compute_graph_distances(A, v)

    shell_profiles = {c: [] for c in [1, 2, 3]}

    for b in range(n_beta):
        for ic in range(n_ic):
            cls = classes[b, ic]
            if cls not in [1, 2, 3]:
                continue

            snap = snapshots[b, ic]
            energy = snap ** 2
            total = np.sum(energy)
            if total < 1e-30:
                continue

            # Find centroid (max energy vertex)
            center = np.argmax(energy)
            distances = distances_cache[center]
            shells = shell_partition(distances)

            # Shell energy fractions
            profile = np.zeros(6)  # 600-cell has diameter 5 -> 6 shells
            for d in range(min(6, len(shells))):
                if d in shells:
                    profile[d] = np.sum(energy[shells[d]]) / total

            shell_profiles[cls].append(profile)

    # Average per class
    shell_means = {}
    shell_stds = {}
    for cls in [1, 2, 3]:
        if shell_profiles[cls]:
            arr = np.array(shell_profiles[cls])
            shell_means[cls] = np.mean(arr, axis=0)
            shell_stds[cls] = np.std(arr, axis=0)
        else:
            shell_means[cls] = np.zeros(6)
            shell_stds[cls] = np.zeros(6)

    return shell_means, shell_stds


def compute_stability_ranking(classes, persistence, ipr, sector_fracs):
    """Compute stability metrics per class."""
    rankings = {}
    for cls in [1, 2, 3]:
        mask = classes.flatten() == cls
        if np.sum(mask) == 0:
            rankings[cls] = {
                'count': 0, 'mean_persistence': 0,
                'mean_ipr': 0, 'mean_backbone_frac': 0,
            }
            continue

        pers = persistence.flatten()[mask]
        iprs = ipr.flatten()[mask]
        sf = sector_fracs.reshape(-1, sector_fracs.shape[-1])[mask]
        n_backbone = min(6, sf.shape[1])
        backbone = np.sum(sf[:, :n_backbone], axis=1)

        rankings[cls] = {
            'count': int(np.sum(mask)),
            'mean_persistence': float(np.mean(pers)),
            'std_persistence': float(np.std(pers)),
            'mean_ipr': float(np.mean(iprs)),
            'mean_backbone_frac': float(np.mean(backbone)),
        }

    return rankings


def main():
    parser = argparse.ArgumentParser(description='Paper IV analysis')
    parser.add_argument('--n_beta', type=int, default=15)
    parser.add_argument('--n_ic', type=int, default=80)
    parser.add_argument('--T', type=float, default=2000)
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--n_controls', type=int, default=3)
    parser.add_argument('--output', type=str, default='results/paper4')
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}"
          f"{' — ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")

    beta_values = np.logspace(-2, np.log10(5.0), args.n_beta)

    # ── H4 graph ──────────────────────────────────────────
    print("\n=== H4 (600-cell) ===")
    A_h4, _ = build_600cell()
    L_h4 = build_laplacian(A_h4).astype(np.float64)
    evals_h4, evecs_h4 = compute_spectrum(L_h4)

    rng = np.random.default_rng(2024)
    t0 = time.time()
    h4_data = run_sweep_p4(
        "H4", A_h4, L_h4, evals_h4, evecs_h4,
        beta_values, args.n_ic, args.batch, args.T, device, rng)
    print(f"  H4 completed in {time.time()-t0:.1f}s")

    # Build coupling matrix
    M_h4, n_stable_h4 = build_coupling_matrix(
        h4_data['sector_fracs'], h4_data['classes'])
    print(f"  Coupling matrix: {n_stable_h4} stable attractors")
    print(f"  Nonzero off-diagonal pairs: "
          f"{np.sum((M_h4 > 0.01) & ~np.eye(len(M_h4), dtype=bool))} / "
          f"{len(M_h4) * (len(M_h4)-1)}")

    # Shell occupancy
    print("  Computing shell occupancy...")
    shell_means, shell_stds = compute_shell_occupancy(
        h4_data['snapshots'], h4_data['classes'], A_h4)

    # Stability ranking
    rankings = compute_stability_ranking(
        h4_data['classes'], h4_data['persistence'],
        h4_data['ipr'], h4_data['sector_fracs'])
    for cls in [1, 2, 3]:
        r = rankings[cls]
        print(f"  Class {cls}: n={r['count']}, "
              f"persistence={r['mean_persistence']:.3f}, "
              f"IPR={r['mean_ipr']:.4f}, "
              f"backbone={r['mean_backbone_frac']:.3f}")

    # ── Control graphs ────────────────────────────────────
    ctrl_coupling = []
    ctrl_beta = beta_values[::3]  # fewer beta for controls

    for m in range(args.n_controls):
        print(f"\n=== Control {m+1}/{args.n_controls} ===")
        if m < args.n_controls - 1:
            A_ctrl = random_regular_graph(120, 12, seed=1000 + m)
            name = f"RRG_{m}"
        else:
            A_ctrl = rewired_graph(A_h4, num_swaps=10000, seed=99)
            name = "Rewired"

        L_ctrl = build_laplacian(A_ctrl).astype(np.float64)
        evals_ctrl, evecs_ctrl = compute_spectrum(L_ctrl)

        rng_ctrl = np.random.default_rng(2024)
        t0 = time.time()
        ctrl_data = run_sweep_p4(
            name, A_ctrl, L_ctrl, evals_ctrl, evecs_ctrl,
            ctrl_beta, args.n_ic // 2, args.batch, args.T // 2,
            device, rng_ctrl)
        print(f"  {name} completed in {time.time()-t0:.1f}s")

        M_ctrl, n_stable_ctrl = build_coupling_matrix(
            ctrl_data['sector_fracs'], ctrl_data['classes'])
        ctrl_coupling.append({
            'name': name,
            'M': M_ctrl,
            'n_stable': n_stable_ctrl,
            'n_sectors': len(ctrl_data['sector_labels']),
        })

    # ── Save results ──────────────────────────────────────
    np.savez(os.path.join(output_dir, 'paper4_results.npz'),
             beta_values=beta_values,
             h4_classes=h4_data['classes'],
             h4_ipr=h4_data['ipr'],
             h4_sector_fracs=h4_data['sector_fracs'],
             h4_persistence=h4_data['persistence'],
             h4_coupling_matrix=M_h4,
             h4_n_stable=n_stable_h4,
             sector_labels_evals=np.array([l[0] for l in h4_data['sector_labels']]),
             sector_labels_mults=np.array([l[1] for l in h4_data['sector_labels']]),
             shell_means_c1=shell_means.get(1, np.zeros(6)),
             shell_means_c2=shell_means.get(2, np.zeros(6)),
             shell_means_c3=shell_means.get(3, np.zeros(6)),
             shell_stds_c1=shell_stds.get(1, np.zeros(6)),
             shell_stds_c2=shell_stds.get(2, np.zeros(6)),
             shell_stds_c3=shell_stds.get(3, np.zeros(6)),
             rankings_c1_persistence=rankings[1]['mean_persistence'],
             rankings_c1_ipr=rankings[1]['mean_ipr'],
             rankings_c1_count=rankings[1]['count'],
             rankings_c2_persistence=rankings[2]['mean_persistence'],
             rankings_c2_ipr=rankings[2]['mean_ipr'],
             rankings_c2_count=rankings[2]['count'],
             rankings_c3_persistence=rankings[3]['mean_persistence'],
             rankings_c3_ipr=rankings[3]['mean_ipr'],
             rankings_c3_count=rankings[3]['count'],
             )

    # Save control coupling matrices
    for i, cc in enumerate(ctrl_coupling):
        np.savez(os.path.join(output_dir, f'ctrl_coupling_{i}.npz'),
                 name=cc['name'], M=cc['M'],
                 n_stable=cc['n_stable'],
                 n_sectors=cc['n_sectors'])

    print(f"\nAll results saved to {output_dir}/")

    # ── Summary statistics for paper ──────────────────────
    n_sectors = M_h4.shape[0]
    n_possible_pairs = n_sectors * (n_sectors - 1) // 2
    n_observed = np.sum(
        (M_h4 + M_h4.T > 0.02) & ~np.eye(n_sectors, dtype=bool)
    ) // 2

    print(f"\n=== PAPER 4 KEY STATISTICS ===")
    print(f"Observed sector pairings: {n_observed} / {n_possible_pairs}")
    print(f"Top 5 pairings:")
    # Get upper triangle
    triu = np.triu_indices(n_sectors, k=1)
    pair_vals = M_h4[triu]
    top5 = np.argsort(pair_vals)[::-1][:5]
    for idx in top5:
        i, j = triu[0][idx], triu[1][idx]
        print(f"  S{i+1}-S{j+1}: {M_h4[i,j]:.3f}")

    backbone_only = np.sum(
        (h4_data['sector_fracs'].reshape(-1, n_sectors)[:, :6].sum(axis=1) > 0.9)
        & (h4_data['classes'].flatten() <= 3)
        & (h4_data['classes'].flatten() >= 1)
    )
    n_total_stable = np.sum(
        (h4_data['classes'].flatten() >= 1) &
        (h4_data['classes'].flatten() <= 3)
    )
    print(f"Backbone-only attractors: {backbone_only}/{n_total_stable} "
          f"({100*backbone_only/max(1,n_total_stable):.1f}%)")


if __name__ == '__main__':
    main()
