"""
GPU-accelerated attractor classification sweep for Paper III.

Runs batches of initial conditions in parallel on GPU using PyTorch.
Each batch shares the same beta value but different ICs.

Usage:
    python paper3_gpu_sweep.py [--n_beta 25] [--n_ic 40] [--T 2000] [--batch 10]
"""

import argparse
import os
import time
import numpy as np
import torch

from build_600cell import build_600cell
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
SAVE_INTERVAL = 200       # save every 200 steps = every 1.0 time units
T_TRANSIENT = 200         # discard first 200 time units
ENERGY_DRIFT_THRESHOLD = 5e-4

# Classification thresholds (calibrated from diagnostic runs)
IPR_DELOCAL = 0.03
IPR_MODERATE = 0.10
IPR_STRONG = 0.08


def integrate_batch_gpu(Phi_batch, Pi_batch, L_gpu, omega0, lam, beta,
                        dt, n_steps, save_interval, device):
    """
    Velocity-Verlet integration for a batch of ICs on GPU.

    Parameters
    ----------
    Phi_batch : tensor (batch, n)
    Pi_batch : tensor (batch, n)
    L_gpu : tensor (n, n)
    Returns dict with numpy arrays.
    """
    w2 = omega0 ** 2
    half_dt = 0.5 * dt
    batch_size, n = Phi_batch.shape

    n_saves = n_steps // save_interval + 1
    times = np.zeros(n_saves)
    Phi_hist = np.zeros((n_saves, batch_size, n))
    energy_hist = np.zeros((n_saves, batch_size))

    Phi = Phi_batch.clone()
    Pi = Pi_batch.clone()

    # Hamiltonian
    def compute_H():
        ke = 0.5 * torch.sum(Pi ** 2, dim=1)
        pe_base = 0.5 * w2 * torch.sum(Phi ** 2, dim=1)
        pe_coup = 0.5 * lam * torch.sum(Phi * (Phi @ L_gpu.T), dim=1)
        pe_nl = 0.25 * beta * torch.sum(Phi ** 4, dim=1)
        return (ke + pe_base + pe_coup + pe_nl).cpu().numpy()

    # Save initial
    Phi_hist[0] = Phi.cpu().numpy()
    energy_hist[0] = compute_H()
    si = 1

    # Initial force: F = -w2*Phi - lam*L@Phi - beta*Phi^3
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
        "Phi_history": Phi_hist[:si],   # (n_saves, batch, n)
        "energy": energy_hist[:si],     # (n_saves, batch)
    }


def generate_ic_batch(n_vertices, eigenvectors, A, batch_size, rng):
    """Generate a batch of diverse initial conditions."""
    Phi_batch = np.zeros((batch_size, n_vertices))
    Pi_batch = np.zeros((batch_size, n_vertices))

    # 8 IC types for better diversity
    ic_types = ['IC1', 'IC2', 'IC3', 'IC4',
                'IC5', 'IC6', 'IC7', 'IC8']

    for b in range(batch_size):
        ic_type = ic_types[b % len(ic_types)]
        amplitude = rng.uniform(0.1, 2.0)

        if ic_type == 'IC1':
            # Single-vertex excitation
            j = rng.integers(0, n_vertices)
            Phi_batch[b, j] = amplitude

        elif ic_type == 'IC2':
            # Random small-amplitude field
            Phi_batch[b] = rng.normal(0, 0.1 * amplitude, n_vertices)

        elif ic_type == 'IC3':
            # Low-mode seed: sectors 1-3 (14 modes)
            n_low = 14
            coeffs = rng.normal(0, amplitude, n_low)
            Phi_batch[b] = eigenvectors[:, :n_low] @ coeffs

        elif ic_type == 'IC4':
            # Local H3 neighbourhood seed
            j = rng.integers(0, n_vertices)
            Phi_batch[b, j] = amplitude
            neighbours = np.where(A[j] > 0)[0]
            Phi_batch[b, neighbours] = amplitude * 0.5

        elif ic_type == 'IC5':
            # Mid-mode seed: sectors 4-6 (77 modes, indices 14-90)
            n_start, n_end = 14, 91
            n_modes = n_end - n_start
            coeffs = rng.normal(0, amplitude * 0.5, n_modes)
            Phi_batch[b] = eigenvectors[:, n_start:n_end] @ coeffs

        elif ic_type == 'IC6':
            # Single backbone mode with perturbation
            k = rng.integers(0, 91)  # backbone mode
            Phi_batch[b] = amplitude * eigenvectors[:, k]
            Phi_batch[b] += 0.05 * amplitude * rng.normal(size=n_vertices)

        elif ic_type == 'IC7':
            # Multi-vertex cluster (3-5 nearby vertices)
            j = rng.integers(0, n_vertices)
            neighbours = np.where(A[j] > 0)[0]
            chosen = rng.choice(neighbours, min(4, len(neighbours)),
                               replace=False)
            Phi_batch[b, j] = amplitude
            Phi_batch[b, chosen] = amplitude * rng.uniform(0.3, 0.8,
                                                            len(chosen))

        elif ic_type == 'IC8':
            # Random broad with momentum
            Phi_batch[b] = rng.normal(0, 0.2 * amplitude, n_vertices)
            Pi_batch[b] = rng.normal(0, 0.1 * amplitude, n_vertices)

    return Phi_batch, Pi_batch


def compute_diagnostics_batch(Phi_hist, energy_hist, times,
                              eigenvectors, eigenvalues, A,
                              n_transient_saves):
    """
    Compute diagnostics for a batch of trajectories.

    Phi_hist: (n_saves, batch, n)
    energy_hist: (n_saves, batch)
    """
    n_saves, batch_size, n = Phi_hist.shape
    dt_save = times[1] - times[0] if len(times) > 1 else 1.0

    results = []

    for b in range(batch_size):
        # Extract post-transient
        Phi_post = Phi_hist[n_transient_saves:, b, :]  # (n_post, n)
        energy_post = energy_hist[n_transient_saves:, b]

        if len(Phi_post) < 10:
            results.append({
                'class': 4, 'ipr_mean': 0.0,
                'n_dominant': 0, 'backbone_frac': 0.0,
                'rec_error': 1.0, 'trap_frac': 0.0,
                'energy_drift': 1.0,
            })
            continue

        # Energy drift
        drift = energy_drift(energy_post)

        if drift > ENERGY_DRIFT_THRESHOLD:
            results.append({
                'class': 4, 'ipr_mean': 0.0,
                'n_dominant': 0, 'backbone_frac': 0.0,
                'rec_error': 1.0, 'trap_frac': 0.0,
                'energy_drift': drift,
            })
            continue

        # IPR
        ipr = inverse_participation_ratio(Phi_post)
        ipr_mean = np.mean(ipr)

        # Spectral composition
        coeffs = Phi_post @ eigenvectors
        sector_E, sector_labels = spectral_sector_energy(
            coeffs, eigenvalues
        )
        sector_E_mean = np.mean(sector_E, axis=0)
        total_sE = np.sum(sector_E_mean)
        if total_sE > 1e-30:
            sector_fracs = sector_E_mean / total_sE
        else:
            sector_fracs = np.zeros(len(sector_labels))

        n_dominant = int(np.sum(sector_fracs > 0.1))
        n_backbone = min(6, len(sector_fracs))
        backbone_frac = float(np.sum(sector_fracs[:n_backbone]))

        # Autocorrelation (fast: subsample if long)
        n_post = len(Phi_post)
        norms = np.linalg.norm(Phi_post, axis=1)
        max_lag = min(n_post // 4, 50)
        max_autocorr = 0.0
        for lag in range(2, max_lag, max(1, max_lag // 25)):
            dots = np.sum(Phi_post[:-lag] * Phi_post[lag:], axis=1)
            norm_prod = norms[:-lag] * norms[lag:]
            mask = norm_prod > 1e-20
            if np.sum(mask) > 0:
                ac = float(np.mean(dots[mask] / norm_prod[mask]))
                if ac > max_autocorr:
                    max_autocorr = ac

        # Simplified energy trapping
        energy_proxy = np.mean(Phi_post ** 2, axis=0)
        total_ep = np.sum(energy_proxy)
        if total_ep > 1e-30:
            # Check 2-neighbourhood for top-energy vertex
            top_v = np.argmax(energy_proxy)
            A2 = A @ A
            nbhd = ((A[top_v] + A2[top_v]) > 0).astype(float)
            nbhd[top_v] = 1.0
            trap_frac = float(np.sum(energy_proxy * nbhd) / total_ep)
        else:
            trap_frac = 0.0

        # Classify
        cls = classify(ipr_mean, n_dominant, backbone_frac,
                       trap_frac, max_autocorr)

        results.append({
            'class': cls, 'ipr_mean': ipr_mean,
            'n_dominant': n_dominant,
            'backbone_frac': backbone_frac,
            'max_autocorr': max_autocorr, 'trap_frac': trap_frac,
            'energy_drift': drift,
        })

    return results


def classify(ipr_mean, n_dominant, backbone_frac, trap_frac, max_autocorr):
    """
    Classify into attractor class 1-4.

    Uses autocorrelation peak (max_autocorr) instead of raw recurrence
    error. High autocorrelation indicates periodic/quasi-periodic behaviour.

    Parameters
    ----------
    ipr_mean : float, time-averaged IPR
    n_dominant : int, sectors with >10% energy
    backbone_frac : float, fraction of energy in backbone sectors
    trap_frac : float, max energy fraction in a 2-neighbourhood
    max_autocorr : float, peak autocorrelation at nonzero lag (0 to 1)
    """
    # Class 3: Breather — high localisation and energy trapping
    if ipr_mean > IPR_STRONG and trap_frac > 0.35:
        return 3

    # Class 3: also catches moderately localised persistent states
    if ipr_mean > 0.05 and trap_frac > 0.4 and max_autocorr > 0.3:
        return 3

    # Class 1: Backbone harmonic — low IPR, single dominant sector
    if ipr_mean < IPR_DELOCAL and n_dominant <= 2 and backbone_frac > 0.7:
        if max_autocorr > 0.2:
            return 1

    # Class 2: Locked multi-mode — moderate structure, multiple sectors
    if (ipr_mean < IPR_MODERATE
            and 2 <= n_dominant <= 6
            and backbone_frac > 0.5):
        return 2

    # Class 2: also low-IPR multi-sector
    if ipr_mean < IPR_DELOCAL and n_dominant >= 2 and backbone_frac > 0.6:
        return 2

    # Class 1: catch remaining low-IPR backbone states
    if ipr_mean < IPR_DELOCAL and backbone_frac > 0.7:
        return 1

    return 4


def run_sweep(n_beta=25, n_ic=40, T=2000, batch_size=10,
              output_dir='results/paper3'):
    """Run the GPU-accelerated attractor sweep."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}"
          f"{' — ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")

    # Build graph
    print("Building 600-cell graph...")
    A, vertices = build_600cell()
    n_vertices = A.shape[0]
    L = build_laplacian(A).astype(np.float64)

    # Eigendecomposition
    eigenvalues, eigenvectors = compute_spectrum(L)
    print(f"Distinct eigenvalues: {len(set(round(float(e), 6) for e in eigenvalues))}")

    L_gpu = torch.tensor(L, dtype=torch.float64, device=device)

    # Beta values
    beta_values = np.logspace(-3, np.log10(5.0), n_beta)
    n_steps = int(T / DT)
    n_transient_saves = int(T_TRANSIENT / (DT * SAVE_INTERVAL))

    print(f"\nSweep: {n_beta} beta x {n_ic} ICs, T={T}, dt={DT}")
    print(f"Steps per trajectory: {n_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {n_beta * (n_ic // batch_size)}")

    # Storage
    all_classes = np.zeros((n_beta, n_ic), dtype=int)
    all_ipr = np.zeros((n_beta, n_ic))
    class_fractions = np.zeros((n_beta, 4))

    # Also store representative configs for figure generation
    representative_configs = {c: None for c in range(1, 5)}
    representative_ipr = {c: 0.0 for c in range(1, 5)}

    total_start = time.time()

    for b_idx, beta in enumerate(beta_values):
        beta_start = time.time()

        for batch_start in range(0, n_ic, batch_size):
            actual_batch = min(batch_size, n_ic - batch_start)

            # Generate ICs
            Phi_np, Pi_np = generate_ic_batch(
                n_vertices, eigenvectors, A, actual_batch, rng
            )

            # Transfer to GPU
            Phi_gpu = torch.tensor(Phi_np, dtype=torch.float64, device=device)
            Pi_gpu = torch.tensor(Pi_np, dtype=torch.float64, device=device)

            # Integrate
            res = integrate_batch_gpu(
                Phi_gpu, Pi_gpu, L_gpu,
                OMEGA0, LAMBDA, float(beta),
                DT, n_steps, SAVE_INTERVAL, device
            )

            # Diagnostics (on CPU)
            diag_list = compute_diagnostics_batch(
                res['Phi_history'], res['energy'], res['times'],
                eigenvectors, eigenvalues, A, n_transient_saves
            )

            for i, diag in enumerate(diag_list):
                ic_idx = batch_start + i
                all_classes[b_idx, ic_idx] = diag['class']
                all_ipr[b_idx, ic_idx] = diag['ipr_mean']

                # Keep representative configs
                cls = diag['class']
                if cls in representative_configs:
                    # Keep the one with most typical IPR for its class
                    if representative_configs[cls] is None:
                        # Take a mid-trajectory snapshot
                        mid = len(res['Phi_history']) // 2
                        representative_configs[cls] = res['Phi_history'][mid, i, :].copy()
                        representative_ipr[cls] = diag['ipr_mean']

        # Compute class fractions
        for c in range(1, 5):
            class_fractions[b_idx, c-1] = np.mean(all_classes[b_idx] == c)

        elapsed = time.time() - beta_start
        print(f"[{b_idx+1:2d}/{n_beta}] beta={beta:.4f} "
              f"C1={class_fractions[b_idx,0]:.2f} "
              f"C2={class_fractions[b_idx,1]:.2f} "
              f"C3={class_fractions[b_idx,2]:.2f} "
              f"C4={class_fractions[b_idx,3]:.2f} "
              f"({elapsed:.1f}s)")

    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # Save results
    np.savez(os.path.join(output_dir, 'phase_diagram.npz'),
             beta_values=beta_values,
             class_fractions=class_fractions)

    np.savez(os.path.join(output_dir, 'attractor_classes.npz'),
             beta_values=beta_values,
             classes=all_classes,
             ipr=all_ipr)

    # Save representative configs
    rep_data = {}
    for c in range(1, 5):
        if representative_configs[c] is not None:
            rep_data[f'class{c}'] = representative_configs[c]
        else:
            rep_data[f'class{c}'] = np.zeros(n_vertices)
    np.savez(os.path.join(output_dir, 'representative_configs.npz'), **rep_data)

    print(f"Results saved to {output_dir}/")
    return beta_values, class_fractions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Paper III GPU attractor sweep'
    )
    parser.add_argument('--n_beta', type=int, default=25,
                        help='Number of beta values')
    parser.add_argument('--n_ic', type=int, default=40,
                        help='Initial conditions per beta')
    parser.add_argument('--T', type=float, default=2000,
                        help='Total simulation time')
    parser.add_argument('--batch', type=int, default=10,
                        help='Batch size for GPU')
    parser.add_argument('--output', type=str, default='results/paper3')
    args = parser.parse_args()

    run_sweep(n_beta=args.n_beta, n_ic=args.n_ic, T=args.T,
              batch_size=args.batch, output_dir=args.output)
