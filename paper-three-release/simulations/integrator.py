"""
Symplectic integrator for the VFD nonlinear oscillator system.

Equation of motion:
    Φ̈_i + ω₀² Φ_i + λ (L Φ)_i + β Φ_i³ = 0

Rewritten as Hamiltonian system:
    Φ̇ = Π
    Π̇ = -ω₀² Φ - λ L Φ - β Φ³

Integrated using velocity-Verlet (symplectic, second-order).
"""

import numpy as np


def force(Phi, L, omega0, lam, beta):
    """
    Compute the force F = -∂H/∂Φ = -ω₀²Φ - λLΦ - βΦ³.

    Parameters
    ----------
    Phi : ndarray (n,), field amplitudes
    L : ndarray (n, n), graph Laplacian
    omega0 : float
    lam : float, coupling constant λ
    beta : float, nonlinearity parameter

    Returns
    -------
    F : ndarray (n,), force vector
    """
    return -omega0**2 * Phi - lam * (L @ Phi) - beta * Phi**3


def hamiltonian(Phi, Pi, L, omega0, lam, beta):
    """
    Compute the Hamiltonian (total energy).

    H = Σ_i [½ Π_i² + ½ω₀² Φ_i² + ¼β Φ_i⁴]
        + ¼λ Σ_{(i,j)∈E} (Φ_i - Φ_j)²

    The coupling term can be written as ½λ Φᵀ L Φ (using the quadratic
    form of the Laplacian, but with the ½ factor from the Hamiltonian).
    """
    kinetic = 0.5 * np.sum(Pi**2)
    potential_base = 0.5 * omega0**2 * np.sum(Phi**2)
    potential_coupling = 0.5 * lam * Phi @ (L @ Phi)
    potential_nonlinear = 0.25 * beta * np.sum(Phi**4)

    return kinetic + potential_base + potential_coupling + potential_nonlinear


def integrate(Phi0, Pi0, L, omega0, lam, beta, dt, n_steps,
              save_interval=10, callback=None):
    """
    Integrate the VFD field equation using velocity-Verlet.

    Parameters
    ----------
    Phi0 : ndarray (n,), initial field
    Pi0 : ndarray (n,), initial momentum (Φ̇)
    L : ndarray (n, n), graph Laplacian
    omega0, lam, beta : float, model parameters
    dt : float, time step
    n_steps : int, total number of steps
    save_interval : int, save state every N steps
    callback : callable or None, called with (step, t, Phi, Pi) at
               each save point

    Returns
    -------
    results : dict with keys:
        'times' : ndarray, saved time points
        'Phi_history' : ndarray (n_saves, n), saved field snapshots
        'Pi_history' : ndarray (n_saves, n), saved momentum snapshots
        'energy' : ndarray (n_saves,), Hamiltonian at each save point
    """
    n = len(Phi0)
    Phi = Phi0.copy()
    Pi = Pi0.copy()

    n_saves = n_steps // save_interval + 1
    times = np.zeros(n_saves)
    Phi_history = np.zeros((n_saves, n))
    Pi_history = np.zeros((n_saves, n))
    energy = np.zeros(n_saves)

    # Save initial state
    save_idx = 0
    times[0] = 0.0
    Phi_history[0] = Phi
    Pi_history[0] = Pi
    energy[0] = hamiltonian(Phi, Pi, L, omega0, lam, beta)

    if callback is not None:
        callback(0, 0.0, Phi, Pi)

    save_idx = 1

    # Velocity-Verlet integration
    F = force(Phi, L, omega0, lam, beta)

    for step in range(1, n_steps + 1):
        # Half-step momentum
        Pi = Pi + 0.5 * dt * F

        # Full-step position
        Phi = Phi + dt * Pi

        # Recompute force at new position
        F = force(Phi, L, omega0, lam, beta)

        # Half-step momentum
        Pi = Pi + 0.5 * dt * F

        # Save if needed
        if step % save_interval == 0 and save_idx < n_saves:
            t = step * dt
            times[save_idx] = t
            Phi_history[save_idx] = Phi
            Pi_history[save_idx] = Pi
            energy[save_idx] = hamiltonian(Phi, Pi, L, omega0, lam, beta)

            if callback is not None:
                callback(step, t, Phi, Pi)

            save_idx += 1

    return {
        "times": times[:save_idx],
        "Phi_history": Phi_history[:save_idx],
        "Pi_history": Pi_history[:save_idx],
        "energy": energy[:save_idx],
    }
