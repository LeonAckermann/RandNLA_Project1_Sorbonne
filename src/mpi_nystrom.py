import numpy as np
from scipy.linalg import cholesky, svd, solve_triangular, qr, eigh
from mpi4py import MPI
from .sketch import generate_sketch_block

def nystrom_mpi(comm, A_local, global_n, k, l, seed=42, sketch='gaussian',
               debug=False, return_timings=False):
    rank = comm.Get_rank()
    size = comm.Get_size()
    q = int(np.sqrt(size))

    # 0. Sanity checks for 2D process grid (P = q^2)
    if q * q != size:
        raise ValueError(f"nystrom_mpi requires P=q^2 processes, got P={size} (q={q})")

    # Helper: synchronize and measure a phase time (global max)
    def _time_phase(world_comm, fn):
        world_comm.Barrier()
        t0 = MPI.Wtime()
        out = fn()
        world_comm.Barrier()
        t1 = MPI.Wtime()
        dt_local = t1 - t0
        dt = world_comm.allreduce(dt_local, op=MPI.MAX)
        return out, dt

    # Helper: measure a row_comm phase and lift to world max
    def _time_row_phase(row_comm, world_comm, fn):
        row_comm.Barrier()
        t0 = MPI.Wtime()
        out = fn()
        row_comm.Barrier()
        t1 = MPI.Wtime()
        dt_local = t1 - t0
        dt_row = row_comm.allreduce(dt_local, op=MPI.MAX)
        dt = world_comm.allreduce(dt_row, op=MPI.MAX)
        return out, dt

    # Helper: measure a col0-only phase (others contribute 0), lifted to world max
    def _time_col0_phase(col_comm, world_comm, active, fn):
        world_comm.Barrier()
        if active:
            col_comm.Barrier()
            t0 = MPI.Wtime()
            out = fn()
            col_comm.Barrier()
            t1 = MPI.Wtime()
            dt_local = t1 - t0
        else:
            out = None
            dt_local = 0.0
        world_comm.Barrier()
        dt = world_comm.allreduce(dt_local, op=MPI.MAX)
        return out, dt

    # Timing dict (all reported as world max)
    timings = {
        "t_total": 0.0,
        "t_sketch": 0.0,
        "t_C_matmul": 0.0,
        "t_C_reduce": 0.0,
        "t_B_matmul": 0.0,
        "t_B_allreduce": 0.0,
        "t_C_gather": 0.0,
        "t_factorization": 0.0,
        "t_parallel_kernel": 0.0,  # C_matmul + C_reduce + B_matmul + B_allreduce
    }

    comm.Barrier()
    t_total_start = MPI.Wtime()

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
    t0 = MPI.Wtime()
    Omega_col = generate_sketch_block(local_cols, l, seed, col_offset, sketch)
    Omega_row = generate_sketch_block(local_rows, l, seed, row_offset, sketch)
    t1 = MPI.Wtime()
    timings["t_sketch"] = comm.allreduce(t1 - t0, op=MPI.MAX)

    # 4. Compute C (Distributed)
    def _do_C_matmul():
        return A_local @ Omega_col

    C_partial, timings["t_C_matmul"] = _time_phase(comm, _do_C_matmul)

    # Sum across the row communicator (each row reduces to its grid_col==0 rank)
    C_row_block = None
    if grid_col == 0:
        C_row_block = np.zeros((local_rows, l), dtype=A_local.dtype)

    def _do_C_reduce():
        row_comm.Reduce(C_partial, C_row_block, op=MPI.SUM, root=0)
        return None

    _, timings["t_C_reduce"] = _time_row_phase(row_comm, comm, _do_C_reduce)

    # 5. Compute B (Replicated) - buffer-based Allreduce (NO pickling)
    def _do_B_matmul():
        return Omega_row.T @ C_partial

    B_partial, timings["t_B_matmul"] = _time_phase(comm, _do_B_matmul)

    def _do_B_allreduce():
        B_global = np.zeros((l, l), dtype=B_partial.dtype)
        mpi_dtype = MPI._typedict[B_partial.dtype.char]
        comm.Allreduce([B_partial, mpi_dtype], [B_global, mpi_dtype], op=MPI.SUM)
        return B_global

    B_global, timings["t_B_allreduce"] = _time_phase(comm, _do_B_allreduce)

    # Stabilize B (numerical symmetry + relative jitter)
    B_global = 0.5 * (B_global + B_global.T)
    eps = np.finfo(B_global.dtype).eps if np.issubdtype(B_global.dtype, np.floating) else 1e-16
    jitter = eps * (np.trace(B_global) / l) if l > 0 else eps
    if jitter == 0.0:
        jitter = 1e-12
    B_global = B_global + jitter * np.eye(l, dtype=B_global.dtype)

    timings["t_parallel_kernel"] = (
        timings["t_C_matmul"] + timings["t_C_reduce"] +
        timings["t_B_matmul"] + timings["t_B_allreduce"]
    )

    # 6. Gather C to Root (buffer-based Gatherv on column 0)
    U_hat, Lambda_k = None, None
    C_global = None

    is_col0 = (grid_col == 0)

    def _do_C_gather():
        nonlocal C_global

        # Only column-0 ranks have C_row_block
        send = C_row_block
        mpi_dtype = MPI._typedict[send.dtype.char]

        # Gather local_rows on col_comm root (small python gather is fine)
        rows_list = col_comm.gather(local_rows, root=0)

        if col_comm.Get_rank() == 0:
            counts = [int(r) * l for r in rows_list]
            displs = [0]
            for cnt in counts[:-1]:
                displs.append(displs[-1] + cnt)
            C_global = np.empty((global_n, l), dtype=send.dtype)
            recv = [C_global.ravel(), counts, displs, mpi_dtype]
        else:
            recv = None

        # Buffer-based Gatherv (NO pickling)
        col_comm.Gatherv([send.ravel(), mpi_dtype], recv, root=0)
        return None

    _, timings["t_C_gather"] = _time_col0_phase(col_comm, comm, is_col0, _do_C_gather)

    # 7. Final Factorization (Rank 0 only)
    def _do_factorization():
        nonlocal U_hat, Lambda_k

        if rank != 0:
            return None

        B = B_global
        Ltri = None
        jitter_local = jitter

        # Progressive jitter for Cholesky
        for _ in range(6):
            try:
                Ltri = cholesky(B + jitter_local * np.eye(l, dtype=B.dtype), lower=True)
                break
            except np.linalg.LinAlgError:
                jitter_local *= 10.0

        if Ltri is not None:
            # Z = C * L^{-T}
            Z = solve_triangular(Ltri, C_global.T, lower=True).T
        else:
            # Fallback (correct): Z = C * B^{-1/2}
            w, U = eigh(B)
            tol = w.max() * 1e-12
            inv_sqrt = np.where(w > tol, 1.0 / np.sqrt(w), 0.0)
            B_inv_sqrt = (U * inv_sqrt) @ U.T
            Z = C_global @ B_inv_sqrt

        # QR then SVD on small matrix
        Q, R_upper = qr(Z, mode='economic')
        Ur, Sigmar, _ = svd(R_upper)

        Ur_k = Ur[:, :k]
        Sigmar_k = Sigmar[:k]

        Lambda_k = Sigmar_k**2
        U_hat = Q @ Ur_k
        return None

    _, timings["t_factorization"] = _time_phase(comm, _do_factorization)

    comm.Barrier()
    t_total_end = MPI.Wtime()
    timings["t_total"] = comm.allreduce(t_total_end - t_total_start, op=MPI.MAX)

    if debug and rank == 0:
        print("[DEBUG] Timings (world max):")
        for key in [
            "t_total", "t_parallel_kernel",
            "t_sketch", "t_C_matmul", "t_C_reduce",
            "t_B_matmul", "t_B_allreduce",
            "t_C_gather", "t_factorization"
        ]:
            print(f"  {key}: {timings[key]:.6f} s")

    if return_timings:
        total_time = timings["t_total"]
        parallel_time = timings["t_parallel_kernel"]
        return U_hat, Lambda_k, total_time, parallel_time, timings

    return U_hat, Lambda_k, timings["t_total"]
