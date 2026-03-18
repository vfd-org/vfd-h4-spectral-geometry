"""
Generate control graphs for comparison with the 600-cell.

All control graphs are 12-regular with 120 vertices but lack H₄ symmetry.
"""

import numpy as np


def random_regular_graph(n=120, d=12, seed=None):
    """
    Generate a random d-regular simple graph on n vertices.

    Uses a greedy stubs-matching approach with local repair:
    starts from a configuration model pairing and fixes any
    self-loops or multi-edges by swapping with valid pairs.

    Parameters
    ----------
    n : int, number of vertices
    d : int, degree of each vertex
    seed : int or None, random seed

    Returns
    -------
    A : ndarray (n, n), adjacency matrix
    """
    rng = np.random.default_rng(seed)

    if (n * d) % 2 != 0:
        raise ValueError("n * d must be even.")

    max_attempts = 1000
    for attempt in range(max_attempts):
        stubs = np.repeat(np.arange(n), d)
        rng.shuffle(stubs)
        pairs = list(stubs.reshape(-1, 2))

        # Build adjacency, tracking problems
        A = np.zeros((n, n), dtype=int)
        good = []
        bad = []
        for idx, (u, v) in enumerate(pairs):
            if u == v or A[u, v] > 0:
                bad.append(idx)
            else:
                A[u, v] += 1
                A[v, u] += 1
                good.append(idx)

        # Try to fix bad pairs by swapping endpoints with good pairs
        repair_rounds = 0
        while bad and repair_rounds < 500:
            repair_rounds += 1
            bi = bad.pop()
            u, v = pairs[bi]

            fixed = False
            rng.shuffle(good)
            for gi in good:
                a, b = pairs[gi]
                # Try swap: (u,v) and (a,b) -> (u,a) and (v,b)
                if u != a and v != b and u != b and v != a:
                    if A[u, a] == 0 and A[v, b] == 0:
                        # Undo old good pair
                        A[a, b] -= 1
                        A[b, a] -= 1
                        # Add new pairs
                        A[u, a] += 1
                        A[a, u] += 1
                        A[v, b] += 1
                        A[b, v] += 1
                        pairs[gi] = (u, a)
                        pairs[bi] = (v, b)
                        good.append(bi)
                        fixed = True
                        break
                    # Try alternative: (u,v) and (a,b) -> (u,b) and (v,a)
                    if A[u, b] == 0 and A[v, a] == 0:
                        A[a, b] -= 1
                        A[b, a] -= 1
                        A[u, b] += 1
                        A[b, u] += 1
                        A[v, a] += 1
                        A[a, v] += 1
                        pairs[gi] = (u, b)
                        pairs[bi] = (v, a)
                        good.append(bi)
                        fixed = True
                        break

            if not fixed:
                bad.append(bi)
                break  # can't fix, retry from scratch

        if not bad and np.all(A.sum(axis=1) == d):
            return A

    raise RuntimeError(
        f"Failed to generate random {d}-regular graph after {max_attempts} "
        "attempts."
    )


def rewired_graph(A_original, num_swaps=5000, seed=None):
    """
    Create a degree-preserving rewired version of a graph.

    Performs edge swaps: if edges (a,b) and (c,d) exist,
    replace with (a,c) and (b,d) if those don't already exist
    and don't create self-loops.

    Parameters
    ----------
    A_original : ndarray, adjacency matrix to rewire
    num_swaps : int, number of swap attempts
    seed : int or None

    Returns
    -------
    A : ndarray, rewired adjacency matrix
    """
    rng = np.random.default_rng(seed)
    A = A_original.copy()
    n = A.shape[0]

    # Get edge list
    edges = list(zip(*np.where(np.triu(A) > 0)))
    num_edges = len(edges)

    swaps_done = 0
    for _ in range(num_swaps):
        # Pick two random edges
        i1, i2 = rng.choice(num_edges, 2, replace=False)
        a, b = edges[i1]
        c, d = edges[i2]

        # Try swap: (a,b),(c,d) -> (a,c),(b,d)
        if rng.random() < 0.5:
            # Alternative swap: (a,b),(c,d) -> (a,d),(b,c)
            c, d = d, c

        # Check no self-loops
        if a == c or b == d:
            continue
        # Check no existing edges
        if A[a, c] > 0 or A[b, d] > 0:
            continue

        # Perform swap
        A[a, b] = A[b, a] = 0
        A[c, d] = A[d, c] = 0
        A[a, c] = A[c, a] = 1
        A[b, d] = A[d, b] = 1

        edges[i1] = (min(a, c), max(a, c))
        edges[i2] = (min(b, d), max(b, d))
        swaps_done += 1

    print(f"  Rewiring: {swaps_done}/{num_swaps} swaps successful")
    return A


def build_control_graphs(A_600cell):
    """
    Generate three control graphs.

    Parameters
    ----------
    A_600cell : ndarray, adjacency matrix of the 600-cell

    Returns
    -------
    controls : dict mapping name -> adjacency matrix
    """
    controls = {}

    print("Generating control graph 1 (random 12-regular)...")
    controls["control1"] = random_regular_graph(120, 12, seed=42)

    print("Generating control graph 2 (random 12-regular)...")
    controls["control2"] = random_regular_graph(120, 12, seed=137)

    print("Generating control graph 3 (rewired 600-cell)...")
    controls["control3"] = rewired_graph(A_600cell, num_swaps=10000, seed=99)

    # Verify all controls
    for name, A in controls.items():
        degrees = A.sum(axis=1)
        n_edges = int(A.sum()) // 2
        sym = np.allclose(A, A.T)
        print(f"  {name}: vertices={A.shape[0]}, "
              f"degree=[{degrees.min()},{degrees.max()}], "
              f"edges={n_edges}, symmetric={sym}")

    return controls


if __name__ == "__main__":
    from build_600cell import build_600cell
    A, _ = build_600cell()
    controls = build_control_graphs(A)
