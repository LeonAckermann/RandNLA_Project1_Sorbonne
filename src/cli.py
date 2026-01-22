import argparse
import time
import datetime
import numpy as np
import os

from .dataset import load_mnist, rbf_kernel, load_year_prediction_msd
from .nystrom import nystrom_rank_k_truncated
from .analysis import save_results, plot_results
from .error import relative_nuclear_error


def run_sweep_sequential(n: int, c: float, l_list: list, k_list: list, sketch: str,
                        dataset: str, dataset_path: str, seed: int, train: bool = True,
                        debug: bool = False):
    """Run sweep experiments sequentially."""
    if debug:
        t_sweep_start = time.time()

    # Load data and build kernel
    if dataset == 'mnist':
        print(f"Loading MNIST {'train' if train else 'test'} set (n={n})")
    else:
        print(f"Loading MSD set (n={n})")

    if debug:
        t_load_start = time.time()

    if dataset == 'mnist':
        X = load_mnist(n=n, train=train)
    else:
        X = load_year_prediction_msd(n=n, path=dataset_path)

    if debug:
        t_load_end = time.time()
        print(f"[DEBUG] Load time: {t_load_end - t_load_start:.4f}s")

    print("Building RBF kernel matrix A")
    if debug:
        t_kernel_start = time.time()

    A = rbf_kernel(X, c)

    if debug:
        t_kernel_end = time.time()
        print(f"[DEBUG] Kernel build time: {t_kernel_end - t_kernel_start:.4f}s, shape: {A.shape}, size: {A.nbytes / (1024**3):.2f} GB")

    nuc_A = np.trace(A)
    print(f"Nuclear norm of A: {nuc_A:.6e}\n")

    results = {}
    for l in l_list:
        print(f"--- Running l={l} ---")
        if debug:
            t_m_start = time.time()

        # Compute Nyström approximation with truncation at max k
        U_approx, S_approx, t_compute = nystrom_rank_k_truncated(
            A, l=l, k=max(k_list), seed=seed, sketch=sketch, debug=debug
        )

        for k in k_list:
            U_k = U_approx[:, :k]
            S_k = S_approx[:k]
            A_approx_k = U_k @ (S_k[:, None] * U_k.T)

            if debug:
                t_k_start = time.time()

            err = relative_nuclear_error(A, A_approx_k, nuc_A=nuc_A, debug=debug)

            results[(l, k)] = {
                'n': n,
                'c': c,
                'l': l,
                'k': k,
                'dataset': dataset,
                'sketch_method': sketch,
                'compute_time': float(t_compute),
                'rel_error': float(err),
                'memory_gb': float(A.nbytes / (1024**3))
            }
            print(f"l={l}, k={k}: Time={t_compute:.4f}s Error={err:.6e}")

            if debug:
                t_k_end = time.time()
                print(f"[DEBUG] Error comp for k={k} time: {t_k_end - t_k_start:.4f}s")

        if debug:
            t_m_end = time.time()
            print(f"[DEBUG] Total for l={l}: {t_m_end - t_m_start:.4f}s")
        print()

    if debug:
        t_sweep_end = time.time()
        print(f"[DEBUG] Total sweep time: {t_sweep_end - t_sweep_start:.4f}s")

    return results


def run_sweep_mpi(n: int, c: float, l_list: list, k_list: list, sketch: str,
                  dataset: str, dataset_path: str, seed: int, train: bool = True,
                  debug: bool = False, summary_csv: str = None, skip_error: bool = False):
    from mpi4py import MPI
    from .mpi_nystrom import nystrom_mpi
    import csv

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    q = int(np.sqrt(size))
    if q * q != size:
        if rank == 0:
            print(f"Error: Size {size} must be a perfect square.")
        comm.Abort(1)

    # -----------------------------
    # 1) Build full kernel only on rank 0 (same as sequential), then distribute blocks
    #    Use buffer-based Send/Recv + chunking to avoid >2GB message issues.
    # -----------------------------
    if rank == 0:
        if dataset == 'mnist':
            X = load_mnist(n=n, train=train)
        else:
            X = load_year_prediction_msd(n=n, path=dataset_path)

        A_full = rbf_kernel(X, c)
        nuc_A = np.trace(A_full)
    else:
        A_full = None
        nuc_A = None

    nuc_A = comm.bcast(nuc_A, root=0)

    # Helper: chunked 2D send/recv using buffer ops
    def _send_block_chunked(dest: int, block: np.ndarray, tag_base: int = 200, max_bytes: int = 512 * 1024 * 1024):
        # Send header (shape + dtype)
        comm.send((block.shape, block.dtype.str), dest=dest, tag=tag_base)

        rows, cols = block.shape
        itemsize = block.dtype.itemsize
        max_elems = max_bytes // itemsize
        rows_per_chunk = max(1, max_elems // max(1, cols))

        mpi_dtype = MPI._typedict[block.dtype.char]
        chunk_idx = 0
        for i0 in range(0, rows, rows_per_chunk):
            i1 = min(rows, i0 + rows_per_chunk)
            chunk = np.ascontiguousarray(block[i0:i1, :])
            comm.Send([chunk, mpi_dtype], dest=dest, tag=tag_base + 1 + chunk_idx)
            chunk_idx += 1

    def _recv_block_chunked(source: int = 0, tag_base: int = 200, max_bytes: int = 512 * 1024 * 1024):
        shape, dtype_str = comm.recv(source=source, tag=tag_base)
        dtype = np.dtype(dtype_str)
        rows, cols = shape
        arr = np.empty((rows, cols), dtype=dtype)

        itemsize = dtype.itemsize
        max_elems = max_bytes // itemsize
        rows_per_chunk = max(1, max_elems // max(1, cols))

        mpi_dtype = MPI._typedict[dtype.char]
        chunk_idx = 0
        for i0 in range(0, rows, rows_per_chunk):
            i1 = min(rows, i0 + rows_per_chunk)
            comm.Recv([arr[i0:i1, :], mpi_dtype], source=source, tag=tag_base + 1 + chunk_idx)
            chunk_idx += 1

        return arr

    # Matrix A distribution (2D grid blocks)
    if rank == 0:
        block_size = n // q
        for r in range(size):
            r_row, r_col = r // q, r % q

            row_s = r_row * block_size
            row_e = (r_row + 1) * block_size if r_row < q - 1 else n
            col_s = r_col * block_size
            col_e = (r_col + 1) * block_size if r_col < q - 1 else n

            block = np.ascontiguousarray(A_full[row_s:row_e, col_s:col_e])

            if r == 0:
                A_local = block
            else:
                _send_block_chunked(dest=r, block=block, tag_base=200 + 10 * r)
    else:
        A_local = _recv_block_chunked(source=0, tag_base=200 + 10 * rank)

    comm.Barrier()

    # Prepare summary CSV (rank 0 only)
    if rank == 0 and summary_csv is not None:
        os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
        file_exists = os.path.exists(summary_csv)
        with open(summary_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "timestamp", "job_id", "dataset", "n", "c", "sketch",
                "P", "q", "l", "k",
                "t_total", "t_parallel_kernel",
                "t_sketch", "t_C_matmul", "t_C_reduce",
                "t_B_matmul", "t_B_allreduce",
                "t_C_gather", "t_factorization",
                "rel_error"
            ])
            if not file_exists:
                w.writeheader()

    # -----------------------------
    # 2) Synchronized Sweep
    # -----------------------------
    results = {}
    for l in l_list:
        if rank == 0:
            print(f"\n--- Running l={l} ---", flush=True)

        max_k = max(k_list)

        U_approx, S_approx, t_total, t_parallel, tb = nystrom_mpi(
            comm=comm,
            A_local=A_local,
            global_n=n,
            k=max_k,
            l=l,
            seed=seed,
            sketch=sketch,
            debug=debug,
            return_timings=True
        )

        if rank == 0:
            for k in k_list:
                U_k = U_approx[:, :k]
                S_k = S_approx[:k]
                A_approx_k = U_k @ (S_k[:, None] * U_k.T)

                if skip_error:
                    err = float("nan")
                else:
                    err = relative_nuclear_error(A_full, A_approx_k, nuc_A=nuc_A)

                results[(l, k)] = {
                    'n': n, 'c': c, 'l': l, 'k': k,
                    'dataset': dataset,
                    'sketch_method': sketch,
                    'compute_time': float(t_total),
                    'parallel_time': float(t_parallel),
                    'rel_error': float(err),
                    'memory_gb': float(A_full.nbytes / (1024**3)) if A_full is not None else 0.0,
                    # breakdown (optional)
                    't_sketch': float(tb["t_sketch"]),
                    't_C_matmul': float(tb["t_C_matmul"]),
                    't_C_reduce': float(tb["t_C_reduce"]),
                    't_B_matmul': float(tb["t_B_matmul"]),
                    't_B_allreduce': float(tb["t_B_allreduce"]),
                    't_C_gather': float(tb["t_C_gather"]),
                    't_factorization': float(tb["t_factorization"]),
                    't_parallel_kernel': float(tb["t_parallel_kernel"]),
                }

                print(f"l={l}, k={k}: Total={t_total:.4f}s (ParKernel={t_parallel:.4f}s) Error={err:.6e}", flush=True)

                if summary_csv is not None:
                    row = {
                        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "job_id": os.getenv("SLURM_JOB_ID", ""),
                        "dataset": dataset,
                        "n": n,
                        "c": c,
                        "sketch": sketch,
                        "P": size,
                        "q": q,
                        "l": l,
                        "k": k,
                        "t_total": float(t_total),
                        "t_parallel_kernel": float(t_parallel),
                        "t_sketch": float(tb["t_sketch"]),
                        "t_C_matmul": float(tb["t_C_matmul"]),
                        "t_C_reduce": float(tb["t_C_reduce"]),
                        "t_B_matmul": float(tb["t_B_matmul"]),
                        "t_B_allreduce": float(tb["t_B_allreduce"]),
                        "t_C_gather": float(tb["t_C_gather"]),
                        "t_factorization": float(tb["t_factorization"]),
                        "rel_error": float(err),
                    }
                    with open(summary_csv, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=row.keys())
                        w.writerow(row)

        comm.Barrier()

    return results if rank == 0 else None


def main():
    parser = argparse.ArgumentParser(description='Randomized Nyström sweep experiments')
    parser.add_argument('--dataset', choices=['mnist', 'msd'], default='msd',
                        help='dataset to use (mnist, msd), default: msd')
    parser.add_argument('--dataset-path', type=str, default='./datasets/YearPredictionMSD.txt',
                        help='path to MSD dataset file')
    parser.add_argument('--n', type=int, required=True, help='number of data rows to use')
    parser.add_argument('--c', type=float, required=True, help='RBF bandwidth parameter')
    parser.add_argument('--l-list', type=str, required=True, help='comma-separated list of sketch sizes l')
    parser.add_argument('--k-list', type=str, required=True, help='comma-separated list of target ranks k')
    parser.add_argument('--sketch', choices=['gaussian', 'srht'], default='gaussian',
                        help='sketching method (default: gaussian)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--train', action='store_true', help='use MNIST training set')
    parser.add_argument('--out-dir', type=str, default='results', help='output directory for results')
    parser.add_argument('--debug', action='store_true', help='enable debug prints with timing')
    parser.add_argument('--mpi', action='store_true', help='run with MPI for distributed computation')

    # NEW: summary CSV for scaling runs
    parser.add_argument('--summary-csv', type=str, default=None,
                        help='append one-line CSV summary (rank 0 only)')
    # NEW: skip expensive error computation
    parser.add_argument('--skip-error', action='store_true',
                        help='skip relative error computation (useful for scaling)')

    args = parser.parse_args()

    l_list = [int(x) for x in args.l_list.split(',') if x.strip()]
    k_list = [int(x) for x in args.k_list.split(',') if x.strip()]

    if not l_list or not k_list:
        raise ValueError('--l-list and --k-list must contain at least one value')

    print(f"Sweep over l={l_list} and k={k_list}")

    if args.mpi:
        results = run_sweep_mpi(
            args.n, args.c, l_list, k_list, args.sketch,
            args.dataset, args.dataset_path, args.seed,
            args.train, args.debug,
            summary_csv=args.summary_csv,
            skip_error=args.skip_error
        )
    else:
        results = run_sweep_sequential(
            args.n, args.c, l_list, k_list, args.sketch,
            args.dataset, args.dataset_path, args.seed,
            args.train, args.debug
        )

    # Save results (rank 0 only for MPI)
    if results is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"{args.out_dir}/{timestamp}"

        nproc = '1'
        if args.mpi:
            nproc = os.getenv('SLURM_NTASKS', os.getenv('OMPI_COMM_WORLD_SIZE', '1'))

        save_results(results, out_dir, meta={
            'dataset': args.dataset,
            'train': args.train,
            'n': args.n,
            'c': args.c,
            'l': l_list,
            'k': k_list,
            'sketch': args.sketch,
            'mpi': args.mpi,
            'seed': args.seed,
            '#proc': nproc,
            'summary_csv': args.summary_csv,
            'skip_error': args.skip_error,
        })

        plot_results(results, out_dir, xlabel='k')
        print(f"Saved results and plots to {out_dir}")


if __name__ == '__main__':
    main()
