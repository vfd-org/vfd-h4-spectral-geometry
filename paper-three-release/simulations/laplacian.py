"""
Laplacian construction and spectral decomposition.
"""

import numpy as np


def build_laplacian(A):
    """
    Compute the graph Laplacian L = dI - A for a d-regular graph.

    Parameters
    ----------
    A : ndarray (n, n), adjacency matrix

    Returns
    -------
    L : ndarray (n, n), graph Laplacian
    """
    d = int(A.sum(axis=1)[0])
    n = A.shape[0]
    L = d * np.eye(n) - A
    return L


def compute_spectrum(L):
    """
    Compute eigenvalues and eigenvectors of a symmetric Laplacian.

    Returns eigenvalues in ascending order.

    Parameters
    ----------
    L : ndarray (n, n), symmetric Laplacian matrix

    Returns
    -------
    eigenvalues : ndarray (n,)
    eigenvectors : ndarray (n, n), columns are eigenvectors
    """
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    return eigenvalues, eigenvectors


def spectral_summary(eigenvalues, tol=1e-6):
    """
    Group eigenvalues by distinct values and report multiplicities.

    Parameters
    ----------
    eigenvalues : ndarray
    tol : float, tolerance for grouping

    Returns
    -------
    distinct : list of (eigenvalue, multiplicity) pairs
    """
    sorted_evals = np.sort(eigenvalues)
    distinct = []
    i = 0
    while i < len(sorted_evals):
        val = sorted_evals[i]
        count = 1
        while i + count < len(sorted_evals) and \
                abs(sorted_evals[i + count] - val) < tol:
            count += 1
        distinct.append((np.mean(sorted_evals[i:i + count]), count))
        i += count
    return distinct


if __name__ == "__main__":
    from build_600cell import build_600cell

    A, _ = build_600cell()
    L = build_laplacian(A)
    evals, evecs = compute_spectrum(L)

    print("\n600-cell Laplacian spectrum:")
    print(f"  Shape: {L.shape}")
    print(f"  Min eigenvalue: {evals[0]:.10f}")
    print(f"  Max eigenvalue: {evals[-1]:.10f}")

    distinct = spectral_summary(evals)
    print(f"\n  Distinct eigenvalues: {len(distinct)}")
    print(f"  {'Eigenvalue':>15s}  {'Multiplicity':>12s}")
    print(f"  {'-'*15}  {'-'*12}")
    for val, mult in distinct:
        print(f"  {val:15.10f}  {mult:12d}")
    print(f"\n  Sum of multiplicities: {sum(m for _, m in distinct)}")
