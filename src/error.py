import numpy as np
from typing import Tuple
import time
from scipy.linalg import cholesky, svd, solve_triangular, qr

def relative_nuclear_error(A: np.ndarray, A_approx: np.ndarray, nuc_A: float = None, debug: bool = False) -> float:
    """
    Computes || A - A_approx ||_* / || A ||_*
    
    Args:
        A: The original SPD matrix (n x n).
        A_approx: The low-rank approximation (n x n).
        nuc_A: Precomputed nuclear norm of A (optional, for speed).
        
    Returns:
        The relative nuclear norm error.
    """
    # 1. Compute Denominator: || A ||_*
    # Optimization: Since A is SPD, Nuclear Norm = Trace(A).
    # This is O(n) instead of O(n^3).
    if nuc_A is None:
        nuc_A = np.trace(A) 
        # Fallback if you want to be strictly generic: 
        # nuc_A = np.sum(np.linalg.svd(A, compute_uv=False))

    # 2. Compute Numerator: || A - A_approx ||_*
    # The difference is likely indefinite, so we MUST use SVD.
    diff = A - A_approx
    if debug:
        print(f"[DEBUG] Computing SVD for nuclear norm of difference matrix of shape {diff.shape}")
        start = time.time()
    singular_values_diff = np.linalg.svd(diff, compute_uv=False)
    if debug:
        end = time.time()
        print(f"[DEBUG] SVD computed in {end - start:.4f}s")
    nuc_diff = np.sum(singular_values_diff)

    return nuc_diff / nuc_A