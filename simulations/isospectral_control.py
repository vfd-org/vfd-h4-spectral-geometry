"""
Isospectral surrogate construction for the 600-cell graph Laplacian.

Builds surrogate Laplacians that preserve eigenvalues but destroy
the H₄ eigenvector geometry. Two variants:

1. Full surrogate: Q_rand Λ Q_rand^T (random orthogonal basis)
2. Block surrogate: randomize within degenerate sectors only
"""

import numpy as np
from laplacian import compute_spectrum, spectral_summary


def random_orthogonal(n, rng):
    """Generate a random orthogonal matrix via QR of Gaussian."""
    M = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(M)
    # Ensure proper orthogonal (fix sign ambiguity)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q


def build_full_surrogate(eigenvalues, n=120, seed=None):
    """
    Build isospectral surrogate with fully randomized eigenvectors.

    L_iso = Q_rand @ diag(eigenvalues) @ Q_rand^T

    Preserves spectrum exactly, destroys all geometric structure.

    Parameters
    ----------
    eigenvalues : ndarray (n,), eigenvalues of the 600-cell Laplacian
    n : int
    seed : int or None

    Returns
    -------
    L_iso : ndarray (n, n)
    Q_rand : ndarray (n, n)
    """
    rng = np.random.default_rng(seed)
    Q_rand = random_orthogonal(n, rng)
    Lambda = np.diag(eigenvalues)
    L_iso = Q_rand @ Lambda @ Q_rand.T
    # Force exact symmetry
    L_iso = 0.5 * (L_iso + L_iso.T)
    return L_iso, Q_rand


def build_block_surrogate(eigenvalues, eigenvectors, n=120, seed=None,
                          tol=1e-6):
    """
    Build isospectral surrogate with block-randomized eigenvectors.

    Within each degenerate eigenvalue sector, apply a random rotation
    to the eigenvectors. Across sectors, eigenvectors are unchanged.

    This preserves eigenspace multiplicities and sector structure
    but randomizes the basis within each sector.

    Parameters
    ----------
    eigenvalues : ndarray (n,)
    eigenvectors : ndarray (n, n), columns are eigenvectors
    seed : int or None
    tol : float, tolerance for grouping eigenvalues

    Returns
    -------
    L_iso : ndarray (n, n)
    Q_block : ndarray (n, n)
    """
    rng = np.random.default_rng(seed)

    sorted_idx = np.argsort(eigenvalues)
    sorted_evals = eigenvalues[sorted_idx]

    # Identify degenerate sectors
    sectors = []
    i = 0
    while i < len(sorted_evals):
        j = i + 1
        while j < len(sorted_evals) and \
                abs(sorted_evals[j] - sorted_evals[i]) < tol:
            j += 1
        sectors.append(sorted_idx[i:j])
        i = j

    Q_block = eigenvectors.copy()

    for indices in sectors:
        k = len(indices)
        if k <= 1:
            continue
        # Random rotation within this sector
        R = random_orthogonal(k, rng)
        # Replace sector eigenvectors with rotated versions
        Q_block[:, indices] = Q_block[:, indices] @ R

    Lambda = np.diag(eigenvalues)
    L_iso = Q_block @ Lambda @ Q_block.T
    L_iso = 0.5 * (L_iso + L_iso.T)
    return L_iso, Q_block


def verify_surrogate(L_original, L_surrogate, name="surrogate"):
    """Verify that surrogate matches original spectrum."""
    evals_orig = np.sort(np.linalg.eigvalsh(L_original))
    evals_surr = np.sort(np.linalg.eigvalsh(L_surrogate))

    max_diff = np.max(np.abs(evals_orig - evals_surr))
    symmetric = np.allclose(L_surrogate, L_surrogate.T)
    psd = evals_surr[0] >= -1e-10

    print(f"  {name}:")
    print(f"    Eigenvalue max diff: {max_diff:.2e}")
    print(f"    Symmetric: {symmetric}")
    print(f"    PSD: {psd}")

    distinct_orig = spectral_summary(evals_orig)
    distinct_surr = spectral_summary(evals_surr)
    print(f"    Distinct eigenvalues: orig={len(distinct_orig)}, "
          f"surr={len(distinct_surr)}")

    return max_diff < 1e-8 and symmetric and psd


if __name__ == "__main__":
    from build_600cell import build_600cell
    from laplacian import build_laplacian

    A, _ = build_600cell()
    L = build_laplacian(A)
    evals, evecs = compute_spectrum(L)

    print("\nBuilding full surrogate...")
    L_full, _ = build_full_surrogate(evals, seed=42)
    verify_surrogate(L, L_full, "full_surrogate")

    print("\nBuilding block surrogate...")
    L_block, _ = build_block_surrogate(evals, evecs, seed=42)
    verify_surrogate(L, L_block, "block_surrogate")
