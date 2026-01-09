"""Sequential randomized Nyström implementations and evaluation utilities."""
from typing import Tuple
import numpy as np
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

def nystrom_column_sample(A: np.ndarray, m: int, k: int, seed: int = None) -> Tuple[np.ndarray, float]:
    """Nyström via uniform column sampling.

    Args:
        A: (n x n) symmetric kernel matrix
        m: number of sampled columns
        k: target rank for truncation
        seed: RNG seed

    Returns:
        A_approx: (n x n) low-rank approximation
        time_taken: seconds
    """
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    idx = rng.choice(n, size=m, replace=False)
    C = A[:, idx]
    W = A[np.ix_(idx, idx)]
    t0 = time.time()
    # SVD of W (small m x m)
    Uw, sw, _ = np.linalg.svd(W)
    # numerical tolerance to treat small singulars
    tol = 1e-12
    pos = sw > tol
    r = max(1, min(k, np.count_nonzero(pos)))
    Uw_k = Uw[:, :r]
    sw_k = sw[:r]
    # form W^+ (pseudo-inverse) restricted to top r
    W_pinv = Uw_k @ np.diag(1.0 / sw_k) @ Uw_k.T
    A_approx = C @ W_pinv @ C.T
    t1 = time.time()
    return A_approx, t1 - t0


def nystrom_gaussian_sketch(A: np.ndarray, m: int, k: int, seed: int = None) -> Tuple[np.ndarray, float]:
    """Nyström-like approximation using a Gaussian sketch Omega (n x m).

    Uses: Y = A @ Omega, Q = orth(Y), B = Q^T A Q, then SVD(B) -> form approx = Q Ub Sigma Ub^T Q^T
    """
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    Omega = rng.normal(size=(n, m))
    t0 = time.time()
    Y = A @ Omega
    Q, _ = np.linalg.qr(Y, mode='reduced')
    B = Q.T @ (A @ Q)
    Ub, sb, _ = np.linalg.svd(B)
    r = min(k, len(sb))
    Ub_k = Ub[:, :r]
    sb_k = sb[:r]
    A_approx = Q @ (Ub_k @ np.diag(sb_k) @ Ub_k.T) @ Q.T
    t1 = time.time()
    return A_approx, t1 - t0


def nystrom_column_full(A: np.ndarray, m: int, seed: int = None, debug: bool = False) -> Tuple[np.ndarray, float]:
    """Nyström reconstruction (no rank truncation) via uniform column sampling.

    Returns (A_nyst, time_taken).
    """
    if debug:
        print(f"[DEBUG] Starting nystrom_column_full: m={m}, A.shape={A.shape}")
        t0 = time.time()
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    idx = rng.choice(n, size=m, replace=False)
    C = A[:, idx]
    W = A[np.ix_(idx, idx)]
    if debug:
        t_sample = time.time()
        print(f"[DEBUG] Sampling time: {t_sample - t0:.4f}s")
    t0_inner = time.time()
    Uw, sw, _ = np.linalg.svd(W)
    tol = 1e-12
    pos = sw > tol
    if np.count_nonzero(pos) == 0:
        W_pinv = np.zeros_like(W)
    else:
        Uw_pos = Uw[:, pos]
        sw_pos = sw[pos]
        W_pinv = Uw_pos @ np.diag(1.0 / sw_pos) @ Uw_pos.T
    if debug:
        t_svd = time.time()
        print(f"[DEBUG] SVD and pinv time: {t_svd - t0_inner:.4f}s")
    A_nyst = C @ W_pinv @ C.T
    t1 = time.time()
    if debug:
        print(f"[DEBUG] Matrix mult time: {t1 - t_svd:.4f}s, total: {t1 - t0:.4f}s")
    return A_nyst, t1 - t0


def nystrom_gaussian_full(A: np.ndarray, m: int, seed: int = None, debug: bool = False) -> Tuple[np.ndarray, float]:
    """Nyström-like reconstruction using Gaussian sketch; returns full reconstruction.
    """
    if debug:
        print(f"[DEBUG] Starting nystrom_gaussian_full: m={m}, A.shape={A.shape}")
        t0 = time.time()
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    Omega = rng.normal(size=(n, m))
    if debug:
        t_omega = time.time()
        print(f"[DEBUG] Omega generation time: {t_omega - t0:.4f}s")
    t0_inner = time.time()
    Y = A @ Omega
    if debug:
        t_y = time.time()
        print(f"[DEBUG] Y = A @ Omega time: {t_y - t0_inner:.4f}s")
    Q, _ = np.linalg.qr(Y, mode='reduced')
    if debug:
        t_qr = time.time()
        print(f"[DEBUG] QR time: {t_qr - t_y:.4f}s")
    B = Q.T @ (A @ Q)
    if debug:
        t_b = time.time()
        print(f"[DEBUG] B = Q.T @ (A @ Q) time: {t_b - t_qr:.4f}s")
    A_nyst = Q @ (B @ Q.T)
    t1 = time.time()
    if debug:
        print(f"[DEBUG] Final mult time: {t1 - t_b:.4f}s, total: {t1 - t0:.4f}s")
    return A_nyst, t1 - t0

def nystrom_rank_k_truncated(A: np.ndarray, k: int, l: int = 10, seed: int = None, debug: bool = False):
    """
    Computes the rank-k truncated Randomized Nyström approximation as per the project slides.
    
    Args:
        A: The symmetric positive semidefinite matrix (n x n).
        k: The target rank for the final truncation.
        p: Oversampling parameter (l = k + p).
        seed: Random seed for reproducibility.
    
    Returns:
        U_hat: The approximate top k eigenvectors (n x k).
        S_squared: The approximate top k eigenvalues (length k).
        timing: Dictionary containing runtime metrics.
    """
    if debug:
        print(f"[DEBUG] Starting Nyström: n={A.shape[0]}, k={k}, p={l - k}")
        t_start = time.time()
    
    n = A.shape[0]
    #l = k + p  # Sketch dimension
    rng = np.random.default_rng(seed)
    
    # 1. Generate Gaussian test matrix Omega
    Omega = rng.normal(size=(n, l))
    
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
    
    if debug:
        t_end = time.time()
        print(f"[DEBUG] Finished in {t_end - t_start:.4f}s")
        
    return U_hat, Lambda_k, (time.time() - t_start)


def save_results(results: dict, out_dir: str, meta: dict = None) -> None:
    """Save sweep results to CSV and JSON metadata.

    `results` should be a dict keyed by (m,k) tuples with values containing
    at least: {'time': float, 'nuc_A': float, 'nuc_A_nyst_k': float, 'rel_error': float}
    """
    import os
    import json
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for (m, k), vals in sorted(results.items()):
        row = {'m': m, 'k': k}
        row.update(vals)
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, 'results.csv')
    df.to_csv(csv_path, index=False)
    if meta is not None:
        with open(os.path.join(out_dir, 'meta.json'), 'w') as jf:
            json.dump(meta, jf, indent=2)


def plot_results(results: dict, out_dir: str, xlabel: str = 'k', title: str = None) -> None:
    """Plot relative error and nuclear norms from `results` and save PNGs in `out_dir`.

    `results` keys are (m,k) tuples; values provide 'rel_error' and 'nuc_A' and 'nuc_A_nyst_k' optionally.
    Generates:
      - rel_error_vs_k.png (one curve per m)
      - nuc_vs_k.png (nuclear norms vs k)
    """
    import os
    import matplotlib.pyplot as plt
    from collections import defaultdict

    os.makedirs(out_dir, exist_ok=True)
    # organize by m
    by_m = defaultdict(list)
    for (m, k), vals in sorted(results.items()):
        by_m[m].append((k, vals))

    # Relative error vs k
    plt.figure()
    for m, items in sorted(by_m.items()):
        items_sorted = sorted(items, key=lambda x: x[0])
        ks = [it[0] for it in items_sorted]
        rels = [it[1].get('rel_error', None) for it in items_sorted]
        plt.plot(ks, rels, marker='o', label=f'm={m}')
    plt.xlabel(xlabel)
    plt.ylabel('Relative nuclear norm error')
    # plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    if title:
        plt.title(title)
    plt.grid(True, which='both', ls=':')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rel_error_vs_k.png'))
    plt.close()

    # Nuclear norms (A_nuc and A_nyst_k if present)
    plt.figure()
    for m, items in sorted(by_m.items()):
        items_sorted = sorted(items, key=lambda x: x[0])
        ks = [it[0] for it in items_sorted]
        nucs = [it[1].get('nuc_A', None) for it in items_sorted]
        plt.plot(ks, nucs, marker='o', label=f'A nuc (m={m})')
        nucs_nyst = [it[1].get('nuc_A_nyst_k', None) for it in items_sorted]
        if any(v is not None for v in nucs_nyst):
            plt.plot(ks, nucs_nyst, marker='x', linestyle='--', label=f'A_nyst_k nuc (m={m})')
    plt.xlabel(xlabel)
    plt.ylabel('Nuclear norm')
    # plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    if title:
        plt.title(title)
    plt.grid(True, which='both', ls=':')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'nuc_vs_k.png'))
    plt.close()
