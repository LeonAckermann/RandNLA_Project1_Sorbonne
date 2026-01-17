import numpy as np
from typing import Tuple
import time
from scipy.linalg import cholesky, svd, solve_triangular, qr

def generate_sketch_block(n_rows: int, l: int, seed: int, row_offset: int, sketch: str) -> np.ndarray:
    """
    Generates a deterministic block of the Gaussian sketching matrix Omega.
    Seeding is offset by the global row index to ensure mathematical consistency 
    across distributed blocks without communication.
    """
    # Create a distinct RNG stream for this block based on its global position
    # (seed + row_offset) ensures that row 0 generated on Proc 0 is identical 
    # to row 0 if it were generated on Proc 1.
    rng = np.random.default_rng(seed + row_offset)
    if sketch == 'gaussian':
        return generate_gaussian_sketch(n_rows, l, rng)
    else: # 'srht'
        return generate_srht_sketch(n_rows, l, rng)

def generate_gaussian_sketch(n: int, l: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a Gaussian random sketching matrix.
    
    Args:
        n: Number of rows.
        l: Number of columns (sketch dimension).
        rng: NumPy random generator instance.
    
    Returns:
        Omega: (n x l) Gaussian random matrix with entries ~ N(0, 1).
    """
    return rng.normal(size=(n, l))


def generate_srht_sketch(n: int, l: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a Subsampled Randomized Hadamard Transform (SRHT) sketching matrix.
    
    The SRHT is computed as Omega = D * H * P where:
    - D is a random diagonal sign matrix (+1/-1)
    - H is the Walsh-Hadamard transform matrix (applied as matrix-free operator)
    - P is a random column subsampling (keeps only l out of n columns)
    
    Args:
        n: Number of rows.
        l: Number of columns (sketch dimension).
        rng: NumPy random generator instance.
    
    Returns:
        Omega: (n x l) SRHT sketching matrix.
    """
    # Ensure n is a power of 2 for Hadamard
    n_padded = 2 ** int(np.ceil(np.log2(n)))
    
    # D: Diagonal random signs
    signs = rng.choice([-1, 1], size=n_padded)
    
    # H: Hadamard transform via fast Walsh-Hadamard transform
    def hadamard_transform(x: np.ndarray) -> np.ndarray:
        """Fast Walsh-Hadamard transform (recursive)."""
        def _fwht(x):
            N = len(x)
            if N == 1:
                return x
            x_even = _fwht(x[0::2])
            x_odd = _fwht(x[1::2])
            return np.concatenate([x_even + x_odd, x_even - x_odd])
        
        N= len(x)
        return _fwht(x) / np.sqrt(N)
    
    # Apply D*H: first sign multiplication, then Hadamard on each row
    DH = np.zeros((n_padded, n_padded))
    for i in range(n_padded):
        row = np.zeros(n_padded)
        row[i] = signs[i]
        DH[i, :] = hadamard_transform(row)
    
    # P: Random column subsampling (keep l columns out of n_padded)
    col_indices = rng.choice(n_padded, size=l, replace=False)
    Omega = DH[:n, :][:, col_indices]
    
    return Omega
