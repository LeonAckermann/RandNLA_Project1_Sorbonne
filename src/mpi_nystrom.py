"""MPI scaffold for distributed Nyström-style computation.

This module provides a simple MPI implementation that:
- broadcasts a randomly generated Omega from root,
- each rank computes its local rows of A (kernel rows for owned row indices),
- computes local Y_local = A_local @ Omega,
- gathers Y at root to proceed with the small-size computations.

This is a pragmatic scaffold intended to match the block-row distribution described
in the project brief. It intentionally keeps logic readable rather than fully
optimized for large-scale runs.
"""
from mpi4py import MPI
import numpy as np
import time
from typing import Tuple
from .dataset import load_mnist, rbf_kernel


def _row_range_for_rank(n: int, rank: int, size: int) -> Tuple[int, int]:
    # simple block row partition
    rows_per = n // size
    rem = n % size
    start = rank * rows_per + min(rank, rem)
    end = start + rows_per + (1 if rank < rem else 0)
    return start, end


def mpi_nystrom(n: int, c: float, m: int, k: int, seed: int = None, train: bool = True):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        X = load_mnist(n=n, train=train)
    else:
        X = None

    # Broadcast full X to all ranks for simplicity (could be made distributed)
    X = comm.bcast(X, root=0)
    n_actual = X.shape[0]
    if n_actual != n:
        n = n_actual

    # Each rank computes its local rows of the kernel: A_local = K[local_rows, :]
    start, end = _row_range_for_rank(n, rank, size)
    X_local = X[start:end, :]
    # compute distances between X_local and X
    sq_norms_local = np.sum(X_local * X_local, axis=1)
    sq_norms = np.sum(X * X, axis=1)
    D2_local = sq_norms_local[:, None] + sq_norms[None, :] - 2.0 * (X_local @ X.T)
    D2_local = np.maximum(D2_local, 0.0)
    A_local = np.exp(-D2_local / (c * c))

    # Root creates Omega (n x m) and broadcasts
    if rank == 0:
        rng = np.random.default_rng(seed)
        Omega = rng.normal(size=(n, m))
    else:
        Omega = None
    Omega = comm.bcast(Omega, root=0)

    t0 = time.time()
    # Local multiply
    Y_local = A_local @ Omega

    # Gather Y_local shapes and data to root
    counts = np.array(comm.gather(Y_local.shape[0], root=0))
    if rank == 0:
        # allocate full Y
        Y = np.empty((n, m), dtype=Y_local.dtype)
        # copy root's block
        Y[start:end, :] = Y_local
        # receive others
        offset = 0
        for r in range(1, size):
            s, e = _row_range_for_rank(n, r, size)
            recv_rows = e - s
            if recv_rows > 0:
                buf = np.empty((recv_rows, m), dtype=Y_local.dtype)
                comm.Recv(buf, source=r, tag=77)
                Y[s:e, :] = buf
    else:
        comm.Send(Y_local, dest=0, tag=77)

    t1 = time.time()

    if rank == 0:
        # root continues sequentially: orthonormalize Y, form B = Q^T A Q, etc.
        Q, _ = np.linalg.qr(Y, mode='reduced')
        # build full A on root (may be memory heavy) and compute B = Q^T A Q
        A_full = rbf_kernel(X, c)
        B = Q.T @ (A_full @ Q)
        Ub, sb, _ = np.linalg.svd(B)
        r = min(k, len(sb))
        Ub_k = Ub[:, :r]
        sb_k = sb[:r]
        A_approx = Q @ (Ub_k @ np.diag(sb_k) @ Ub_k.T) @ Q.T
        return A_approx, float(t1 - t0)
    else:
        return None, float(t1 - t0)


if __name__ == '__main__':
    # small local test (not intended to be run under mpirun here)
    raise RuntimeError("This module is meant to be invoked via src.cli under mpirun")
