import numpy as np
import time
from scipy.linalg import cholesky, svd, solve_triangular, qr, eigh

from .sketch import generate_gaussian_sketch, generate_srht_sketch


def nystrom_rank_k_truncated(
    A: np.ndarray,
    k: int,
    l: int = 10,
    seed: int = None,
    sketch: str = 'gaussian',
    debug: bool = False
):
    if debug:
        print(f"[DEBUG] Starting Nyström: n={A.shape[0]}, k={k}, l={l}, sketch={sketch}")
        t_start = time.time()

    if sketch not in ['gaussian', 'srht']:
        raise ValueError(f"sketch must be 'gaussian' or 'srht', got '{sketch}'")

    n = A.shape[0]
    rng = np.random.default_rng(seed)

    # Generate Omega
    if sketch == 'gaussian':
        Omega = generate_gaussian_sketch(n, l, rng)
    else:
        Omega = generate_srht_sketch(n, l, rng)

    if debug:
        print(f"[DEBUG] Sketch matrix Omega generated: shape {Omega.shape}")

    t_compute_start = time.time()

    # C = A * Omega, B = Omega^T * C
    C = A @ Omega
    B = Omega.T @ C

    # Symmetrize + jitter (relative)
    B = 0.5 * (B + B.T)
    eps = np.finfo(B.dtype).eps if np.issubdtype(B.dtype, np.floating) else 1e-16
    jitter = eps * (np.trace(B) / l) if l > 0 else eps
    if jitter == 0.0:
        jitter = 1e-12
    B = B + jitter * np.eye(l, dtype=B.dtype)

    if debug:
        print(f"[DEBUG] Sketches computed. Shape C: {C.shape}, Shape B: {B.shape}")

    # Step 1-2: Build Z = C * B^{-1/2}
    # Preferred: Cholesky => B = L L^T, then Z = C * L^{-T}
    # Fallback (correct): eig => B^{-1/2} and Z = C * B^{-1/2}
    Ltri = None
    jitter_local = jitter
    for _ in range(6):
        try:
            Ltri = cholesky(B + jitter_local * np.eye(l, dtype=B.dtype), lower=True)
            break
        except np.linalg.LinAlgError:
            jitter_local *= 10.0

    if Ltri is not None:
        Z = solve_triangular(Ltri, C.T, lower=True).T
    else:
        # Correct fallback: Z = C * B^{-1/2}
        w, U = eigh(B)
        tol = w.max() * 1e-12
        inv_sqrt = np.where(w > tol, 1.0 / np.sqrt(w), 0.0)
        B_inv_sqrt = (U * inv_sqrt) @ U.T
        Z = C @ B_inv_sqrt

    # Step 3: QR of Z
    Q, R_upper = qr(Z, mode='economic')

    # Step 4: SVD of R (small)
    Ur, Sigmar, _ = svd(R_upper)

    Ur_k = Ur[:, :k]
    Sigmar_k = Sigmar[:k]

    Lambda_k = Sigmar_k ** 2
    U_hat = Q @ Ur_k

    t_compute_end = time.time()
    compute_time = t_compute_end - t_compute_start

    if debug:
        t_end = time.time()
        print(f"[DEBUG] Finished in {t_end - t_start:.4f}s (U_hat/Lambda_k computation: {compute_time:.4f}s)")

    return U_hat, Lambda_k, compute_time
