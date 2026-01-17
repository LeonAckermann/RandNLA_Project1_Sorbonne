import numpy as np
import time
import sys
from scipy.linalg import cholesky, svd, solve_triangular, qr
from mpi4py import MPI
from .sketch import generate_sketch_block 

def nystrom_mpi(comm, A_local, global_n, k, l, seed=42, sketch='gaussian', debug=False):
    rank = comm.Get_rank()
    size = comm.Get_size()
    q = int(np.sqrt(size))
    
    # 1. Detect actual local dimensions
    local_rows, local_cols = A_local.shape
    
    grid_row = rank // q
    grid_col = rank % q
    
    row_comm = comm.Split(color=grid_row, key=grid_col)
    col_comm = comm.Split(color=grid_col, key=grid_row)

    # 2. Synchronize row/col offsets
    row_offset = col_comm.scan(local_rows) - local_rows
    col_offset = row_comm.scan(local_cols) - local_cols

    # 3. Generate Local Sketch Blocks
    Omega_col = generate_sketch_block(local_cols, l, seed, col_offset, sketch)
    Omega_row = generate_sketch_block(local_rows, l, seed, row_offset, sketch)

    comm.Barrier()
    
    # 4. Compute C (Distributed)
    # C = A * Omega
    C_partial = A_local @ Omega_col
    
    # Sum across the row communicator
    C_row_block = None
    if grid_col == 0:
        C_row_block = np.zeros((local_rows, l), dtype=A_local.dtype)
    
    row_comm.Reduce(C_partial, C_row_block, op=MPI.SUM, root=0)

    # 5. Compute B (Replicated)
    # B = Omega.T * A * Omega = Omega.T * C
    B_partial = Omega_row.T @ C_partial
    B_global = comm.allreduce(B_partial, op=MPI.SUM)
    B_global += np.eye(l) * 1e-12

    # 6. Gather C to Root
    U_hat, Lambda_k, total_time = None, None, 0.0
    
    row_counts = col_comm.allgather(local_rows) if grid_col == 0 else None

    C_global = None
    if rank == 0:
        C_global = np.zeros((global_n, l), dtype=A_local.dtype)
        t_start = time.time()

    if grid_col == 0:
        all_C_blocks = col_comm.gather(C_row_block, root=0)
        if rank == 0:
            C_global = np.vstack(all_C_blocks)

    # 7. Final Factorization (Rank 0 only)
    if rank == 0:
        # Step 1: Cholesky Decomposition of B -> B = LL^T
        # If B is rank deficient, use SVD/Eig instead.
        try:
            L = cholesky(B_global, lower=True)
        except np.linalg.LinAlgError:
            if debug: print("[DEBUG] Cholesky failed in MPI, falling back to SVD")
            Ub, Sb, _ = svd(B_global)
            L = Ub @ np.diag(np.sqrt(Sb))

        # Step 2: Z = C * L^{-T}
        # We solve L * Z.T = C.T for Z.T, then transpose back
        Z = solve_triangular(L, C_global.T, lower=True).T
        
        # Step 3: QR Factorization of Z -> Z = QR
        Q, R_upper = qr(Z, mode='economic')
        
        # Step 4: Truncated SVD of R -> R = Uk * Sigma_k * Vk.T
        Ur, Sigmar, _ = svd(R_upper)
        
        # Truncate to top k components
        Ur_k = Ur[:, :k]
        Sigmar_k = Sigmar[:k]
        
        # Final Factorization Construction
        # Eigenvalues are Sigma_k^2
        Lambda_k = Sigmar_k**2
        
        # Eigenvectors U_hat = Q * Ur_k
        U_hat = Q @ Ur_k
        
        total_time = time.time() - t_start

    comm.Barrier()
    return U_hat, Lambda_k, total_time