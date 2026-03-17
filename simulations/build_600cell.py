"""
Build the 600-cell graph from the binary icosahedral group.

Generates 120 vertices on S^3 ⊂ R^4, determines adjacency by
edge-length criterion, and returns the adjacency matrix.
"""

import numpy as np
from itertools import product


def _even_permutations_4(v):
    """Return the 12 even permutations of a 4-vector."""
    # The 12 even permutations of (a,b,c,d) are the even elements of S_4.
    even_perms = [
        (0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2),
        (1, 0, 3, 2), (1, 2, 0, 3), (1, 3, 2, 0),
        (2, 0, 1, 3), (2, 1, 3, 0), (2, 3, 0, 1),
        (3, 0, 2, 1), (3, 1, 0, 2), (3, 2, 1, 0),
    ]
    results = []
    for p in even_perms:
        results.append(np.array([v[p[0]], v[p[1]], v[p[2]], v[p[3]]]))
    return results


def generate_vertices():
    """
    Generate the 120 vertices of the 600-cell inscribed in the unit 3-sphere.

    Construction follows the binary icosahedral group decomposition:
      - 8 vertices: permutations of (±1, 0, 0, 0)
      - 16 vertices: (±1/2, ±1/2, ±1/2, ±1/2)
      - 96 vertices: even permutations of (0, ±1, ±φ, ±φ̂)/2
        where φ = (1+√5)/2, φ̂ = (1-√5)/2
    """
    phi = (1 + np.sqrt(5)) / 2
    phi_hat = (1 - np.sqrt(5)) / 2

    vertices = set()

    def add(v):
        # Round to avoid floating-point duplicates
        key = tuple(np.round(v, 10))
        vertices.add(key)

    # Group 1: 8 vertices — all permutations of (±1, 0, 0, 0)
    for i in range(4):
        for s in [1, -1]:
            v = np.zeros(4)
            v[i] = s
            add(v)

    # Group 2: 16 vertices — (±1/2, ±1/2, ±1/2, ±1/2)
    for signs in product([0.5, -0.5], repeat=4):
        add(np.array(signs))

    # Group 3: 96 vertices — even permutations of (0, ±1, ±φ, ±φ̂)/2
    base_values = [0, 1, phi, phi_hat]
    # Generate all sign combinations for the non-zero entries
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            for s3 in [1, -1]:
                base = np.array([0, s1 * 1, s2 * phi, s3 * phi_hat]) / 2
                for pv in _even_permutations_4(base):
                    add(pv)

    vertex_array = np.array(sorted(vertices))

    if len(vertex_array) != 120:
        raise ValueError(
            f"Expected 120 vertices, got {len(vertex_array)}. "
            "Check vertex generation."
        )

    # Verify all lie on unit sphere
    norms = np.linalg.norm(vertex_array, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-10):
        raise ValueError("Not all vertices lie on the unit sphere.")

    return vertex_array


def build_adjacency(vertices, epsilon=1e-6):
    """
    Build the adjacency matrix of the 600-cell graph.

    Two vertices are adjacent iff their inner product equals φ/2,
    where φ = (1+√5)/2. This corresponds to edge length 1/φ.

    Parameters
    ----------
    vertices : ndarray of shape (120, 4)
    epsilon : float, tolerance for inner product comparison

    Returns
    -------
    A : ndarray of shape (120, 120), binary adjacency matrix
    """
    phi = (1 + np.sqrt(5)) / 2
    target_ip = phi / 2  # ≈ 0.80902

    n = len(vertices)
    gram = vertices @ vertices.T
    A = (np.abs(gram - target_ip) < epsilon).astype(int)
    np.fill_diagonal(A, 0)

    return A


def verify_graph(A):
    """Verify the 600-cell graph properties."""
    n = A.shape[0]
    degrees = A.sum(axis=1)

    checks = {
        "num_vertices": n == 120,
        "all_degree_12": np.all(degrees == 12),
        "symmetric": np.allclose(A, A.T),
        "num_edges": int(A.sum()) // 2 == 720,
    }

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    if not all(checks.values()):
        raise ValueError("600-cell graph verification failed.")

    return checks


def build_600cell():
    """
    Main entry point: build and verify the 600-cell graph.

    Returns
    -------
    A : ndarray (120, 120), adjacency matrix
    vertices : ndarray (120, 4), vertex coordinates
    """
    vertices = generate_vertices()
    A = build_adjacency(vertices)
    print("600-cell graph verification:")
    verify_graph(A)
    return A, vertices


if __name__ == "__main__":
    A, vertices = build_600cell()
    print(f"\nAdjacency matrix shape: {A.shape}")
    print(f"Vertex array shape: {vertices.shape}")
    print(f"Degree distribution: min={A.sum(1).min()}, max={A.sum(1).max()}")
    print(f"Total edges: {int(A.sum()) // 2}")
