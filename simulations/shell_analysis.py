"""
Distance-shell analysis on the 600-cell graph.

Computes graph distances from a source vertex and tracks
energy propagation through distance shells.
"""

import numpy as np
from collections import deque


def compute_graph_distances(A, source=0):
    """
    BFS to compute shortest-path distances from source vertex.

    Parameters
    ----------
    A : ndarray (n, n), adjacency matrix
    source : int, source vertex

    Returns
    -------
    distances : ndarray (n,), distance from source to each vertex
    """
    n = A.shape[0]
    distances = -np.ones(n, dtype=int)
    distances[source] = 0
    queue = deque([source])

    while queue:
        v = queue.popleft()
        for u in np.where(A[v] > 0)[0]:
            if distances[u] == -1:
                distances[u] = distances[v] + 1
                queue.append(u)

    return distances


def shell_partition(distances):
    """
    Partition vertices by distance shell.

    Returns
    -------
    shells : dict mapping distance -> array of vertex indices
    """
    shells = {}
    for d in range(distances.max() + 1):
        shells[d] = np.where(distances == d)[0]
    return shells


def shell_energy(Phi_history, shells):
    """
    Compute energy in each distance shell over time.

    E_d(t) = Σ_{i: dist(i)=d} Φ_i(t)²

    Parameters
    ----------
    Phi_history : ndarray (n_times, n)
    shells : dict from shell_partition

    Returns
    -------
    shell_E : dict mapping distance -> ndarray (n_times,)
    """
    shell_E = {}
    for d, indices in shells.items():
        shell_E[d] = np.sum(Phi_history[:, indices]**2, axis=1)
    return shell_E


def shell_summary(A, source=0):
    """
    Print shell structure summary for a graph.

    Parameters
    ----------
    A : adjacency matrix
    source : source vertex
    """
    distances = compute_graph_distances(A, source)
    shells = shell_partition(distances)

    print(f"  Shell structure from vertex {source}:")
    for d in sorted(shells.keys()):
        print(f"    distance {d} → {len(shells[d])} vertices")
    print(f"    total: {sum(len(s) for s in shells.values())}")

    return distances, shells
