"""
Graph-invariant classifier validation for Paper III fairness check.

Re-classifies all trajectories using ONLY graph-invariant metrics
(IPR, persistence, energy variance) — no eigenmode projections.

Also runs a rewired 600-cell as an additional structured control.

Usage:
    python paper3_invariant_validation.py
"""

import os
import numpy as np
import torch
import time

from build_600cell import build_600cell
from build_control_graphs import random_regular_graph, rewired_graph
from laplacian import build_laplacian, compute_spectrum
from diagnostics import inverse_participation_ratio, energy_drift


# ── Parameters (identical to baseline) ────────────────────
OMEGA0 = 1.0
LAMBDA = 1.0
DT = 0.005
SAVE_INTERVAL = 200
T_TRANSIENT = 200
ENERGY_DRIFT_THRESHOLD = 5e-4


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


def generate_ic_batch(n_vertices, A, batch_size, rng):
    """Generate ICs without eigenvector dependence."""
    Phi_batch = np.zeros((batch_size, n_vertices))
    Pi_batch = np.zeros((batch_size, n_vertices))
    ic_types = ['single', 'random', 'cluster', 'momentum',
                'single', 'random', 'cluster', 'momentum']

    for b in range(batch_size):
        ic_type = ic_types[b % len(ic_types)]
        amplitude = rng.uniform(0.1, 2.0)

        if ic_type == 'single':
            j = rng.integers(0, n_vertices)
            Phi_batch[b, j] = amplitude
        elif ic_type == 'random':
            Phi_batch[b] = rng.normal(0, 0.1 * amplitude, n_vertices)
        elif ic_type == 'cluster':
            j = rng.integers(0, n_vertices)
            Phi_batch[b, j] = amplitude
            neighbours = np.where(A[j] > 0)[0]
            chosen = rng.choice(neighbours,
                               min(4, len(neighbours)), replace=False)
            Phi_batch[b, chosen] = amplitude * rng.uniform(0.3, 0.8,
                                                            len(chosen))
        elif ic_type == 'momentum':
            Phi_batch[b] = rng.normal(0, 0.2 * amplitude, n_vertices)
            Pi_batch[b] = rng.normal(0, 0.1 * amplitude, n_vertices)

    return Phi_batch, Pi_batch


def classify_invariant(ipr_mean, persistence, energy_variance_ratio,
                       trap_frac):
    """
    Graph-invariant classifier. No eigenmode projections.

    Uses only:
      - IPR (spatial localisation)
      - persistence (temporal stability)
      - energy variance ratio (spatial heterogeneity)
      - trap_frac (energy trapping in 2-neighbourhood)
    """
    # Class 3: Localised — high IPR + trapping
    if ipr_mean > 0.05 and trap_frac > 0.35:
        return 3
    if ipr_mean > 0.08:
        return 3

    # Class 1: Extended harmonic — low IPR + high persistence
    if ipr_mean < 0.03 and persistence > 0.5 and energy_variance_ratio < 0.5:
        return 1

    # Class 2: Multi-mode — intermediate IPR + moderate persistence
    if ipr_mean < 0.10 and persistence > 0.3:
        return 2

    # Class 1: catch remaining low-IPR persistent states
    if ipr_mean < 0.03 and persistence > 0.3:
        return 1

    # Class 4: Transitional
    return 4


def compute_persistence(Phi_post, norms):
    """Autocorrelation-based persistence (no eigenbasis)."""
    n_post = len(Phi_post)
    max_lag = min(n_post // 4, 50)
    max_ac = 0.0
    for lag in range(2, max_lag, max(1, max_lag // 25)):
        dots = np.sum(Phi_post[:-lag] * Phi_post[lag:], axis=1)
        norm_prod = norms[:-lag] * norms[lag:]
        mask = norm_prod > 1e-20
        if np.sum(mask) > 0:
            ac = float(np.mean(dots[mask] / norm_prod[mask]))
            max_ac = max(max_ac, ac)
    return max_ac


def run_graph_invariant(name, A, L, beta_values, n_ic, batch_size,
                        T, device, rng):
    """Run sweep with graph-invariant classification only."""
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
                n_vertices, A, actual_batch, rng)
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
                    continue

                drift = energy_drift(energy_post)
                if drift > ENERGY_DRIFT_THRESHOLD:
                    continue

                # Graph-invariant metrics only
                ipr = inverse_participation_ratio(Phi_post)
                ipr_mean = float(np.mean(ipr))
                all_ipr[b_idx, ic_idx] = ipr_mean

                norms = np.linalg.norm(Phi_post, axis=1)
                persistence = compute_persistence(Phi_post, norms)

                # Energy variance ratio
                per_vertex_E = np.mean(Phi_post ** 2, axis=0)
                total_E = np.sum(per_vertex_E)
                if total_E > 1e-30:
                    ev_ratio = float(np.std(per_vertex_E) / np.mean(per_vertex_E))
                else:
                    ev_ratio = 0.0

                # Energy trapping
                if total_E > 1e-30:
                    top_v = np.argmax(per_vertex_E)
                    A2 = A @ A
                    nbhd = ((A[top_v] + A2[top_v]) > 0).astype(float)
                    nbhd[top_v] = 1.0
                    trap_frac = float(np.sum(per_vertex_E * nbhd) / total_E)
                else:
                    trap_frac = 0.0

                cls = classify_invariant(ipr_mean, persistence,
                                         ev_ratio, trap_frac)
                class_fractions[b_idx, cls - 1] += 1

        row_total = np.sum(class_fractions[b_idx])
        if row_total > 0:
            class_fractions[b_idx] /= row_total

    return class_fractions, all_ipr


def main():
    output_dir = 'results/paper3_invariant'
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}"
          f"{' — ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")

    beta_values = np.array([0.001, 0.005, 0.01, 0.05, 0.1,
                            0.5, 1.0, 2.0, 3.5, 5.0])
    n_ic = 40
    batch_size = 10
    T = 500
    n_controls = 5
    n_beta = len(beta_values)

    # ── H4 ────────────────────────────────────────────────
    print("\n=== H4 (graph-invariant classifier) ===")
    A_h4, _ = build_600cell()
    L_h4 = build_laplacian(A_h4).astype(np.float64)

    rng = np.random.default_rng(42)
    t0 = time.time()
    h4_fracs, h4_ipr = run_graph_invariant(
        "H4", A_h4, L_h4, beta_values, n_ic, batch_size, T, device, rng)
    print(f"  Completed in {time.time()-t0:.1f}s")
    for b_idx, beta in enumerate(beta_values):
        print(f"  beta={beta:.3f}: C1={h4_fracs[b_idx,0]:.2f} "
              f"C2={h4_fracs[b_idx,1]:.2f} C3={h4_fracs[b_idx,2]:.2f} "
              f"C4={h4_fracs[b_idx,3]:.2f}")

    # ── RRG controls ──────────────────────────────────────
    print(f"\n=== Random 12-regular (invariant, M={n_controls}) ===")
    ctrl_fracs_all = np.zeros((n_controls, n_beta, 4))
    for m in range(n_controls):
        print(f"  Control {m+1}/{n_controls}")
        A_ctrl = random_regular_graph(120, 12, seed=1000+m)
        L_ctrl = build_laplacian(A_ctrl).astype(np.float64)
        rng_ctrl = np.random.default_rng(42)
        t0 = time.time()
        ctrl_f, _ = run_graph_invariant(
            f"RRG_{m}", A_ctrl, L_ctrl, beta_values,
            n_ic, batch_size, T, device, rng_ctrl)
        ctrl_fracs_all[m] = ctrl_f
        print(f"    Completed in {time.time()-t0:.1f}s")

    ctrl_mean = np.mean(ctrl_fracs_all, axis=0)
    ctrl_std = np.std(ctrl_fracs_all, axis=0)

    # ── Rewired 600-cell ──────────────────────────────────
    print("\n=== Rewired 600-cell (additional control) ===")
    A_rewired = rewired_graph(A_h4, num_swaps=10000, seed=99)
    L_rewired = build_laplacian(A_rewired).astype(np.float64)
    evals_rew, _ = compute_spectrum(L_rewired)
    n_distinct_rew = len(set(round(float(e), 6) for e in evals_rew))
    print(f"  Distinct eigenvalues: {n_distinct_rew}")

    rng_rew = np.random.default_rng(42)
    t0 = time.time()
    rew_fracs, rew_ipr = run_graph_invariant(
        "Rewired", A_rewired, L_rewired, beta_values,
        n_ic, batch_size, T, device, rng_rew)
    print(f"  Completed in {time.time()-t0:.1f}s")
    for b_idx, beta in enumerate(beta_values):
        print(f"  beta={beta:.3f}: C1={rew_fracs[b_idx,0]:.2f} "
              f"C2={rew_fracs[b_idx,1]:.2f} C3={rew_fracs[b_idx,2]:.2f} "
              f"C4={rew_fracs[b_idx,3]:.2f}")

    # ── Summary ───────────────────────────────────────────
    print("\n=== INVARIANT CLASSIFIER SUMMARY ===")
    print(f"H4:      C1={np.mean(h4_fracs[:,0]):.3f} "
          f"C2={np.mean(h4_fracs[:,1]):.3f} "
          f"C3={np.mean(h4_fracs[:,2]):.3f} "
          f"C4={np.mean(h4_fracs[:,3]):.3f}")
    print(f"RRG:     C1={np.mean(ctrl_mean[:,0]):.3f} "
          f"C2={np.mean(ctrl_mean[:,1]):.3f} "
          f"C3={np.mean(ctrl_mean[:,2]):.3f} "
          f"C4={np.mean(ctrl_mean[:,3]):.3f}")
    print(f"Rewired: C1={np.mean(rew_fracs[:,0]):.3f} "
          f"C2={np.mean(rew_fracs[:,1]):.3f} "
          f"C3={np.mean(rew_fracs[:,2]):.3f} "
          f"C4={np.mean(rew_fracs[:,3]):.3f}")

    # Separation check
    h4_structured = np.mean(h4_fracs[:, :3].sum(axis=1))
    rrg_structured = np.mean(ctrl_mean[:, :3].sum(axis=1))
    rew_structured = np.mean(rew_fracs[:, :3].sum(axis=1))
    print(f"\nFraction structured (C1+C2+C3):")
    print(f"  H4:      {h4_structured:.3f}")
    print(f"  RRG:     {rrg_structured:.3f}")
    print(f"  Rewired: {rew_structured:.3f}")
    print(f"  Separation H4 vs RRG: {h4_structured - rrg_structured:.3f}")

    np.savez(os.path.join(output_dir, 'invariant_results.npz'),
             beta_values=beta_values,
             h4_fracs=h4_fracs, h4_ipr=h4_ipr,
             ctrl_mean=ctrl_mean, ctrl_std=ctrl_std,
             ctrl_fracs_all=ctrl_fracs_all,
             rew_fracs=rew_fracs, rew_ipr=rew_ipr,
             n_distinct_rewired=n_distinct_rew)
    print(f"\nSaved to {output_dir}/")


if __name__ == '__main__':
    main()
