"""
GPU-accelerated Velocity-Verlet integrator using PyTorch CUDA.

Runs the full VFD nonlinear oscillator system on GPU:
    Φ̈ + ω₀²Φ + λLΦ + βΦ³ = 0

For 120×120 systems with 500K steps, this is ~50-100× faster than NumPy.
"""

import numpy as np
import torch


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("  WARNING: CUDA not available, using CPU")
    return dev


def integrate_gpu(Phi0, Pi0, L_np, omega0, lam, beta, dt, n_steps,
                  save_interval=100, device=None):
    """
    Velocity-Verlet integration on GPU.

    Parameters
    ----------
    Phi0 : ndarray (n,), initial field
    Pi0 : ndarray (n,), initial momentum
    L_np : ndarray (n, n), Laplacian matrix (dense)
    omega0, lam, beta : float, model parameters
    dt : float, time step
    n_steps : int, total steps
    save_interval : int, save every N steps
    device : torch.device or None

    Returns
    -------
    dict with 'times', 'Phi_history', 'energy' as numpy arrays
    """
    if device is None:
        device = get_device()

    n = len(Phi0)
    w2 = omega0 ** 2

    # Transfer to GPU
    Phi = torch.tensor(Phi0, dtype=torch.float64, device=device)
    Pi = torch.tensor(Pi0, dtype=torch.float64, device=device)
    L = torch.tensor(L_np, dtype=torch.float64, device=device)

    n_saves = n_steps // save_interval + 1
    # Keep save buffers on CPU to avoid GPU memory growth
    times_np = np.zeros(n_saves)
    Phi_hist_np = np.zeros((n_saves, n))
    energy_np = np.zeros(n_saves)

    half_dt = 0.5 * dt

    def compute_H():
        kinetic = 0.5 * torch.dot(Pi, Pi)
        pot_base = 0.5 * w2 * torch.dot(Phi, Phi)
        pot_coup = 0.5 * lam * torch.dot(Phi, L @ Phi)
        pot_nl = 0.25 * beta * torch.sum(Phi ** 4)
        return (kinetic + pot_base + pot_coup + pot_nl).item()

    # Save initial state
    Phi_hist_np[0] = Phi.cpu().numpy()
    energy_np[0] = compute_H()
    si = 1

    # Initial force
    F = -w2 * Phi - lam * (L @ Phi) - beta * Phi * Phi * Phi

    for step in range(1, n_steps + 1):
        Pi.add_(F, alpha=half_dt)
        Phi.add_(Pi, alpha=dt)
        F = -w2 * Phi - lam * (L @ Phi) - beta * Phi * Phi * Phi
        Pi.add_(F, alpha=half_dt)

        if step % save_interval == 0 and si < n_saves:
            times_np[si] = step * dt
            Phi_hist_np[si] = Phi.cpu().numpy()
            energy_np[si] = compute_H()
            si += 1

    return {
        "times": times_np[:si],
        "Phi_history": Phi_hist_np[:si],
        "energy": energy_np[:si],
    }


def benchmark(L_np, n_steps=10000, device=None):
    """Quick benchmark to show GPU speedup."""
    import time
    if device is None:
        device = get_device()

    n = L_np.shape[0]
    Phi0 = np.zeros(n); Phi0[0] = 1.0
    Pi0 = np.zeros(n)

    t0 = time.time()
    res = integrate_gpu(Phi0, Pi0, L_np, 1.0, 1.0, 0.5, 0.01, n_steps,
                        save_interval=n_steps, device=device)
    elapsed = time.time() - t0
    print(f"  {n_steps} steps in {elapsed:.2f}s "
          f"({n_steps/elapsed:.0f} steps/s)")
    drift = abs(res["energy"][-1] - res["energy"][0]) / \
            (abs(res["energy"][0]) + 1e-30)
    print(f"  Energy drift: {drift:.2e}")
    return elapsed


if __name__ == "__main__":
    from build_600cell import build_600cell
    from laplacian import build_laplacian

    A, _ = build_600cell()
    L = build_laplacian(A).astype(np.float64)

    print("\nGPU benchmark:")
    device = get_device()
    benchmark(L, n_steps=50000, device=device)

    print("\nCPU benchmark:")
    benchmark(L, n_steps=50000, device=torch.device("cpu"))
