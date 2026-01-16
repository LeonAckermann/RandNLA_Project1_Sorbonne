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
    # Instead of global_n // q, use what we actually have in memory
    local_rows, local_cols = A_local.shape
    
    grid_row = rank // q
    grid_col = rank % q
    
    row_comm = comm.Split(color=grid_row, key=grid_col)
    col_comm = comm.Split(color=grid_col, key=grid_row)

    # 2. Synchronize row/col offsets
    # Every rank needs to know where its block starts in the global matrix
    # We use scan to find the offset based on actual block sizes
    row_offset = col_comm.scan(local_rows) - local_rows
    col_offset = row_comm.scan(local_cols) - local_cols

    # 3. Generate Local Sketch Blocks
    # Now Omega_col matches exactly the number of columns in A_local
    Omega_col = generate_sketch_block(local_cols, l, seed, col_offset, sketch)
    Omega_row = generate_sketch_block(local_rows, l, seed, row_offset, sketch)

    comm.Barrier()
    
    # 4. Compute C (Distributed)
    # C_partial is (local_rows x l)
    C_partial = A_local @ Omega_col
    
    # Sum across the row communicator
    C_row_block = None
    if grid_col == 0:
        C_row_block = np.zeros((local_rows, l), dtype=A_local.dtype)
    
    row_comm.Reduce(C_partial, C_row_block, op=MPI.SUM, root=0)

    # 5. Compute B (Replicated)
    B_partial = Omega_row.T @ C_partial
    B_global = comm.allreduce(B_partial, op=MPI.SUM)
    B_global += np.eye(l) * 1e-12

    # 6. Gather C to Root
    U_hat, Lambda_k, total_time = None, None, 0.0
    
    # To use Gatherv (which handles different block sizes), 
    # Rank 0 needs to know how many rows each processor in the column has.
    row_counts = col_comm.allgather(local_rows) if grid_col == 0 else None

    C_global = None
    if rank == 0:
        C_global = np.zeros((global_n, l), dtype=A_local.dtype)
        t_start = time.time()

    if grid_col == 0:
        # We use the lowercase gather for simplicity with mismatched shapes
        # This collects the list of (local_rows x l) blocks
        all_C_blocks = col_comm.gather(C_row_block, root=0)
        if rank == 0:
            C_global = np.vstack(all_C_blocks)

    # 7. Final Factorization (Rank 0 only)
    if rank == 0:
        # ... (Same SVD/QR logic as before) ...
        L = cholesky(B_global, lower=True)
        Z = solve_triangular(L, C_global.T, lower=True).T
        Q, R_upper = qr(Z, mode='economic')
        Ur, Sigmar, _ = svd(R_upper)
        Lambda_k = Sigmar[:k]**2
        U_hat = Q @ Ur[:, :k]
        total_time = time.time() - t_start

    comm.Barrier()
    return U_hat, Lambda_k, total_time