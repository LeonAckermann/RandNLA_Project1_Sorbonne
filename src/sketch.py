import numpy as np

def generate_sketch_block(n_rows: int, l: int, seed: int, row_offset: int, sketch: str) -> np.ndarray:
    """
    Deterministic block generator for Omega (n_rows x l) using global row_offset.
    """
    rng = np.random.default_rng(seed + int(row_offset))
    if sketch == 'gaussian':
        return generate_gaussian_sketch(n_rows, l, rng)
    if sketch == 'srht':
        return generate_srht_sketch(n_rows, l, rng)
    raise ValueError(f"sketch must be 'gaussian' or 'srht', got '{sketch}'")


def generate_gaussian_sketch(n: int, l: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(n, l))


def _fwht_inplace(X: np.ndarray) -> None:
    """
    In-place Fast Walsh–Hadamard Transform on the first axis (rows).
    X shape: (N, l). N must be a power of 2.
    """
    N = X.shape[0]
    h = 1
    while h < N:
        step = h * 2
        for i in range(0, N, step):
            a = X[i:i+h, :]
            b = X[i+h:i+step, :]
            X[i:i+h, :] = a + b
            X[i+h:i+step, :] = a - b
        h = step
    X *= (1.0 / np.sqrt(N))


def generate_srht_sketch(n: int, l: int, rng: np.random.Generator) -> np.ndarray:
    """
    Matrix-free SRHT sketch Omega = sqrt(n_padded/l) * (D H P) restricted to first n rows.

    Returns Omega of shape (n, l). This still materializes Omega (as any method that returns Omega must),
    but avoids the catastrophic O(n^2) DH construction.
    """
    if l > n:
        raise ValueError(f"SRHT requires l <= n, got l={l}, n={n}")

    n_padded = 1 << int(np.ceil(np.log2(max(1, n))))
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=n_padded, replace=True)

    # P: choose l columns (i.e., l basis vectors e_j)
    col_idx = rng.choice(n_padded, size=l, replace=False)

    # Build E = P as n_padded x l with one-hot rows at col_idx
    X = np.zeros((n_padded, l), dtype=np.float64)
    X[col_idx, np.arange(l)] = 1.0

    # Apply H to those l vectors at once (FWHT along rows)
    _fwht_inplace(X)

    # Apply D (row-wise signs)
    X *= signs[:, None]

    # Restrict to first n rows and apply SRHT scaling
    Omega = X[:n, :]
    Omega *= np.sqrt(n_padded / float(l))
    return Omega
