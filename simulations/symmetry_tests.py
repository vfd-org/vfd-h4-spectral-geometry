"""
Symmetry equivalence tests for the 600-cell graph.

Verifies that H₄-equivalent vertices produce statistically
identical dynamics under the VFD field equation.
"""

import numpy as np


def find_equivalent_vertex(A, v0=0):
    """
    Find a vertex that is related to v0 by the symmetry of the graph.

    For a vertex-transitive graph like the 600-cell, all vertices
    are equivalent. We pick a vertex that is NOT a neighbor of v0
    and NOT v0 itself, to ensure the initial conditions are
    spatially separated.

    Parameters
    ----------
    A : ndarray (n, n), adjacency matrix
    v0 : int, reference vertex

    Returns
    -------
    v1 : int, equivalent vertex (non-adjacent to v0)
    """
    n = A.shape[0]
    neighbors = set(np.where(A[v0] > 0)[0])
    neighbors.add(v0)

    # Pick the first non-neighbor
    for v in range(n):
        if v not in neighbors:
            return v

    raise ValueError("All vertices are neighbors of v0")


def compare_trajectories(ipr_0, ipr_1):
    """
    Compare two IPR trajectories for statistical equivalence.

    Parameters
    ----------
    ipr_0, ipr_1 : ndarray, IPR time series

    Returns
    -------
    stats : dict with comparison metrics
    """
    mean_diff = abs(np.mean(ipr_0) - np.mean(ipr_1))
    std_diff = abs(np.std(ipr_0) - np.std(ipr_1))
    correlation = np.corrcoef(ipr_0, ipr_1)[0, 1]

    return {
        "mean_ipr_v0": float(np.mean(ipr_0)),
        "mean_ipr_v1": float(np.mean(ipr_1)),
        "mean_difference": float(mean_diff),
        "std_difference": float(std_diff),
        "correlation": float(correlation),
        "equivalent": bool(mean_diff < 0.05 and abs(correlation) < 0.5),
    }


def detect_breathers(ipr, times, threshold=0.3, min_duration_steps=5):
    """
    Detect breather-like events: contiguous intervals where IPR > threshold.

    Parameters
    ----------
    ipr : ndarray, IPR time series
    times : ndarray, corresponding times
    threshold : float, IPR threshold
    min_duration_steps : int, minimum duration in saved steps

    Returns
    -------
    breathers : list of dicts with keys:
        'start_time', 'end_time', 'duration', 'peak_ipr', 'peak_time'
    """
    above = ipr > threshold
    breathers = []

    i = 0
    while i < len(above):
        if above[i]:
            start = i
            while i < len(above) and above[i]:
                i += 1
            end = i
            duration = end - start

            if duration >= min_duration_steps:
                segment = ipr[start:end]
                peak_idx = start + np.argmax(segment)
                breathers.append({
                    "start_time": float(times[start]),
                    "end_time": float(times[end - 1]),
                    "duration": float(times[end - 1] - times[start]),
                    "peak_ipr": float(ipr[peak_idx]),
                    "peak_time": float(times[peak_idx]),
                    "duration_steps": int(duration),
                })
        else:
            i += 1

    return breathers
