"""Sequential randomized Nyström implementations and evaluation utilities."""
from typing import Tuple
import numpy as np
import time
from scipy.linalg import cholesky, svd, solve_triangular, qr

from .sketch import generate_gaussian_sketch, generate_srht_sketch

def nystrom_rank_k_truncated(A: np.ndarray, k: int, l: int = 10, seed: int = None, sketch: str = 'gaussian', debug: bool = False):
    """
    Computes the rank-k truncated Randomized Nyström approximation as per the project slides.
    
    Args:
        A: The symmetric positive semidefinite matrix (n x n).
        k: The target rank for the final truncation.
        l: Sketch dimension (oversampling parameter, typically l = k + p where p is oversampling).
        seed: Random seed for reproducibility.
        sketch: Type of sketching matrix ('gaussian' or 'srht'). Default: 'gaussian'.
        debug: Enable debug prints with timing information.
    
    Returns:
        U_hat: The approximate top k eigenvectors (n x k).
        Lambda_k: The approximate top k eigenvalues (length k).
        t: Elapsed time for the computation.
    """
    if debug:
        print(f"[DEBUG] Starting Nyström: n={A.shape[0]}, k={k}, l={l}, sketch={sketch}")
        t_start = time.time()
    
    if sketch not in ['gaussian', 'srht']:
        raise ValueError(f"sketch must be 'gaussian' or 'srht', got '{sketch}'")
    
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    
    # 1. Generate test matrix Omega based on sketch type
    if sketch == 'gaussian':
        Omega = generate_gaussian_sketch(n, l, rng)
    else:  # 'srht'
        Omega = generate_srht_sketch(n, l, rng)
    
    if debug:
        print(f"[DEBUG] Sketch matrix Omega generated: shape {Omega.shape}")
    
    t_compute_start = time.time()
    # 2. Compute Sketch Matrices C and B
    # C = A * Omega
    C = A @ Omega
    # B = Omega.T * A * Omega = Omega.T * C
    B = Omega.T @ C
    
    # Add small jitter for stability if needed (optional, standard in Nystrom)
    B = B + np.eye(l) * 1e-12 

    if debug:
        print(f"[DEBUG] Sketches computed. Shape C: {C.shape}, Shape B: {B.shape}")

    # 3. Step 1: Cholesky Decomposition of B -> B = LL^T
    # Slide Remark: If B is rank deficient, use SVD/Eig instead. 
    # We use a try-except block to handle this robustness.
    try:
        L = cholesky(B, lower=True)
    except np.linalg.LinAlgError:
        if debug: print("[DEBUG] Cholesky failed, falling back to SVD for B")
        Ub, Sb, _ = svd(B)
        L = Ub @ np.diag(np.sqrt(Sb))

    # 4. Step 2: Whitening -> Z = C * L^{-T}
    # We solve L * Z.T = C.T for Z.T, then transpose back
    Z = solve_triangular(L, C.T, lower=True).T
    
    # 5. Step 3: QR Factorization of Z -> Z = QR
    Q, R_upper = qr(Z, mode='economic')
    
    # 6. Step 4: Truncated SVD of R -> R = Uk * Sigma_k * Vk.T
    # Note: R is small (l x l), so this SVD is fast.
    
    Ur, Sigmar, Vrt = svd(R_upper)
    
    # Truncate to top k components
    Ur_k = Ur[:, :k]
    Sigmar_k = Sigmar[:k]
    
    # 7. Final Factorization Construction
    # Eigenvalues are Sigma_k^2
    Lambda_k = Sigmar_k**2
    
    # Eigenvectors U_hat = Q * Ur_k
    U_hat = Q @ Ur_k
    t_compute_end = time.time()
    
    compute_time = t_compute_end - t_compute_start
    
    if debug:
        t_end = time.time()
        print(f"[DEBUG] Finished in {t_end - t_start:.4f}s (U_hat/Lambda_k computation: {compute_time:.4f}s)")
        
    return U_hat, Lambda_k, compute_time


