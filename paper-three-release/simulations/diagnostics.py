"""
Diagnostic metrics for VFD simulation analysis.
"""

import numpy as np


def inverse_participation_ratio(Phi):
    """
    Compute the Inverse Participation Ratio (IPR).

    IPR = Σ Φ_i⁴ / (Σ Φ_i²)²

    IPR → 1/n for uniform distribution (delocalized)
    IPR → 1   for single-site concentration (localized)

    Parameters
    ----------
    Phi : ndarray (n,) or (n_times, n)

    Returns
    -------
    ipr : float or ndarray
    """
    if Phi.ndim == 1:
        s2 = np.sum(Phi**2)
        if s2 < 1e-30:
            return 0.0
        return np.sum(Phi**4) / s2**2

    # Batch computation
    s2 = np.sum(Phi**2, axis=1)
    s4 = np.sum(Phi**4, axis=1)
    mask = s2 > 1e-30
    ipr = np.zeros(len(Phi))
    ipr[mask] = s4[mask] / s2[mask]**2
    return ipr


def spectral_coefficients(Phi, eigenvectors):
    """
    Project the field onto the Laplacian eigenbasis.

    c_k(t) = ⟨Φ(t), v_k⟩

    Parameters
    ----------
    Phi : ndarray (n,) or (n_times, n)
    eigenvectors : ndarray (n, n), columns are eigenvectors

    Returns
    -------
    coeffs : ndarray (n,) or (n_times, n)
    """
    if Phi.ndim == 1:
        return Phi @ eigenvectors
    return Phi @ eigenvectors


def spectral_sector_energy(coeffs, eigenvalues, tol=1e-6):
    """
    Compute energy in each degenerate spectral sector.

    Groups eigenvalues by distinct value and sums |c_k|² within
    each group.

    Parameters
    ----------
    coeffs : ndarray (n_times, n), spectral coefficients
    eigenvalues : ndarray (n,)
    tol : float

    Returns
    -------
    sector_energies : ndarray (n_times, n_sectors)
    sector_labels : list of (eigenvalue, multiplicity) pairs
    """
    sorted_indices = np.argsort(eigenvalues)
    sorted_evals = eigenvalues[sorted_indices]

    # Identify sectors
    sectors = []
    i = 0
    while i < len(sorted_evals):
        val = sorted_evals[i]
        j = i + 1
        while j < len(sorted_evals) and abs(sorted_evals[j] - val) < tol:
            j += 1
        sectors.append((np.mean(sorted_evals[i:j]), sorted_indices[i:j]))
        i = j

    n_times = coeffs.shape[0] if coeffs.ndim == 2 else 1
    if coeffs.ndim == 1:
        coeffs = coeffs.reshape(1, -1)

    sector_energies = np.zeros((n_times, len(sectors)))
    sector_labels = []

    for s_idx, (val, indices) in enumerate(sectors):
        sector_energies[:, s_idx] = np.sum(coeffs[:, indices]**2, axis=1)
        sector_labels.append((val, len(indices)))

    return sector_energies, sector_labels


def recurrence(Phi_history, max_lag=None):
    """
    Compute autocorrelation of the field configuration.

    C(τ) = ⟨Φ(t), Φ(t+τ)⟩ / (‖Φ(t)‖ ‖Φ(t+τ)‖)

    Averaged over all valid t.

    Parameters
    ----------
    Phi_history : ndarray (n_times, n)
    max_lag : int or None, maximum lag in saved steps

    Returns
    -------
    lags : ndarray
    correlation : ndarray
    """
    n_times = Phi_history.shape[0]
    if max_lag is None:
        max_lag = min(n_times // 2, 500)

    norms = np.linalg.norm(Phi_history, axis=1)
    mask = norms > 1e-30

    correlation = np.zeros(max_lag)
    counts = np.zeros(max_lag)

    for tau in range(max_lag):
        for t in range(n_times - tau):
            if mask[t] and mask[t + tau]:
                c = np.dot(Phi_history[t], Phi_history[t + tau])
                c /= norms[t] * norms[t + tau]
                correlation[tau] += c
                counts[tau] += 1

        if counts[tau] > 0:
            correlation[tau] /= counts[tau]

    lags = np.arange(max_lag)
    return lags, correlation


def energy_drift(energy):
    """
    Compute relative energy drift over the simulation.

    Parameters
    ----------
    energy : ndarray (n_times,)

    Returns
    -------
    max_relative_drift : float
    """
    E0 = energy[0]
    if abs(E0) < 1e-30:
        return 0.0
    return np.max(np.abs(energy - E0)) / abs(E0)


def compute_all_diagnostics(results, eigenvectors, eigenvalues):
    """
    Compute all diagnostics from simulation results.

    Parameters
    ----------
    results : dict from integrator.integrate()
    eigenvectors : ndarray (n, n)
    eigenvalues : ndarray (n,)

    Returns
    -------
    diag : dict of diagnostic arrays
    """
    Phi_hist = results["Phi_history"]
    times = results["times"]
    energy = results["energy"]

    ipr = inverse_participation_ratio(Phi_hist)
    coeffs = spectral_coefficients(Phi_hist, eigenvectors)
    sector_E, sector_labels = spectral_sector_energy(
        coeffs, eigenvalues
    )
    lags, corr = recurrence(Phi_hist, max_lag=min(len(times) // 2, 500))
    drift = energy_drift(energy)

    return {
        "times": times,
        "ipr": ipr,
        "spectral_coefficients": coeffs,
        "sector_energies": sector_E,
        "sector_labels": sector_labels,
        "recurrence_lags": lags,
        "recurrence_correlation": corr,
        "energy": energy,
        "energy_drift": drift,
    }
