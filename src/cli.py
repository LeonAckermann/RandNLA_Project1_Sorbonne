import argparse
import time
import datetime
from typing import Optional
import numpy as np
import os
from .dataset import load_mnist, rbf_kernel
from .nystrom import (
    nystrom_column_sample,
    nystrom_gaussian_sketch,
    nystrom_rank_k_truncated,
    relative_nuclear_error,
    # full variants + utilities
    nystrom_column_full,
    nystrom_gaussian_full,

    save_results,
    plot_results,
)



def run_sequential(n: int, c: float, m: int, k: int, method: str, seed: Optional[int], train: bool = True, debug: bool = False, approx_nuclear: bool = False):
    if debug:
        print(f"[DEBUG] Starting sequential run: n={n}, c={c}, m={m}, k={k}, method={method}")
        t_start = time.time()
    print(f"Loading MNIST {'train' if train else 'test'} set (n={n})")
    if debug:
        t_load_start = time.time()
    X = load_mnist(n=n, train=train)
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
    print(f"Kernel matrix shape: {A.shape}")
    if method == 'colsample':
        A_approx, t = nystrom_column_sample(A, m=m, k=k, seed=seed)
    elif method == 'gaussian':
        A_approx, t = nystrom_gaussian_sketch(A, m=m, k=k, seed=seed)
    else:
        raise ValueError(f"Unknown method: {method}")
    rel_err = relative_nuclear_error(A, A_approx, approx=approx_nuclear)
    print(f"Method: {method}, sample m={m}, rank k={k}")
    print(f"Time (s): {t:.4f}")
    print(f"Relative nuclear norm error: {rel_err:.6e}")
    if debug:
        t_end = time.time()
        print(f"[DEBUG] Total time: {t_end - t_start:.4f}s")
    return {'time': t, 'rel_err': rel_err}


def run_mpi(n: int, c: float, m: int, k: int, seed: Optional[int], train: bool = True, debug: bool = False, approx_nuclear: bool = False):
    if debug:
        print(f"[DEBUG] Starting MPI run: n={n}, c={c}, m={m}, k={k}")
        t_start = time.time()
    # Import inside function to avoid MPI import on non-mpi runs
    from .mpi_nystrom import mpi_nystrom

    print("Running MPI Nyström scaffold (start mpirun with desired ranks)")
    A_approx, t = mpi_nystrom(n=n, c=c, m=m, k=k, seed=seed, train=train)
    if A_approx is not None:
        print(f"MPI root computed approximation; time local mult (s): {t:.4f}")
        # compute error against full A (root only)
        if debug:
            t_error_start = time.time()
        X = load_mnist(n=n, train=train)
        A = rbf_kernel(X, c)
        if debug:
            t_error_end = time.time()
            print(f"[DEBUG] Error computation setup time: {t_error_end - t_error_start:.4f}s")
        rel_err = relative_nuclear_error(A, A_approx, approx=approx_nuclear)
        print(f"Relative nuclear norm error (MPI root): {rel_err:.6e}")
        if debug:
            t_end = time.time()
            print(f"[DEBUG] Total time: {t_end - t_start:.4f}s")
        return {'time': t, 'rel_err': rel_err}
    else:
        print("Non-root MPI ranks finished local work.")
        return {'time': t}


def main():
    parser = argparse.ArgumentParser(description='Randomized Nyström experiment CLI')
    parser.add_argument('--mode', choices=['sequential', 'mpi'], default='sequential')
    parser.add_argument('--train', action='store_true', help='use MNIST training set (default); use --no-train for test set')
    parser.add_argument('--n', type=int, required=True, help='number of data rows to use')
    parser.add_argument('--c', type=float, required=True, help='RBF bandwidth parameter')
    parser.add_argument('--m', type=int, required=False, help='sketch/sample size (not required for --sweep)')
    parser.add_argument('--k', type=int, required=False, help='target rank k for truncation (not required for --sweep)')
    parser.add_argument('--method', choices=['colsample', 'gaussian'], default='colsample')
    parser.add_argument('--seed', type=int, default=0)
    # sweep options
    parser.add_argument('--sweep', action='store_true', help='run sweep over m and k')
    parser.add_argument('--m-list', type=str, default='', help='comma-separated list of sketch sizes m for sweep')
    parser.add_argument('--k-list', type=str, default='', help='comma-separated list of target ranks k for sweep')
    parser.add_argument('--out-dir', type=str, default='results', help='output directory for sweep results')
    parser.add_argument('--debug', action='store_true', help='enable debug prints with timing')
    parser.add_argument('--approx-nuclear', action='store_true', help='use approximate nuclear norm (randomized SVD) for speed')
    parser.add_argument('--save-eigvals', action='store_true', help='save eigenvalues of A to file for reuse')
    parser.add_argument('--load-eigvals', action='store_true', help='load eigenvalues of A from file instead of recomputing')
    args = parser.parse_args()

    if args.sweep:
        # parse lists
        if not args.m_list or not args.k_list:
            raise ValueError('For --sweep you must provide --m-list and --k-list')
        m_list = [int(x) for x in args.m_list.split(',') if x.strip()]
        k_list = [int(x) for x in args.k_list.split(',') if x.strip()]
    if args.sweep:
        # parse lists
        if not args.m_list or not args.k_list:
            raise ValueError('For --sweep you must provide --m-list and --k-list')
        m_list = [int(x) for x in args.m_list.split(',') if x.strip()]
        k_list = [int(x) for x in args.k_list.split(',') if x.strip()]
        print(f"Sweep over m={m_list} and k={k_list}")
        if args.debug:
            t_sweep_start = time.time()
        # load data and build kernel once
        print(f"Loading MNIST {'train' if args.train else 'test'} set (n={args.n})")
        if args.debug:
            t_load_start = time.time()
        X = load_mnist(n=args.n, train=args.train)
        if args.debug:
            t_load_end = time.time()
            print(f"[DEBUG] Load time: {t_load_end - t_load_start:.4f}s")
        print("Building RBF kernel matrix A")
        if args.debug:
            t_kernel_start = time.time()
        A = rbf_kernel(X, args.c)
        if args.debug:
            t_kernel_end = time.time()
            print(f"[DEBUG] Kernel build time: {t_kernel_end - t_kernel_start:.4f}s, shape: {A.shape}, size: {A.nbytes / (1024**3):.2f} GB")

        # Compute or load nuclear norm of A
        #eigvals_file = f"eigvals_n{args.n}_c{args.c}_train{args.train}.npy"
        #if args.load_eigvals and os.path.exists(eigvals_file):
        #    print(f"Loading eigenvalues from {eigvals_file}")
        #    eigvals = np.load(eigvals_file)
        #    nuc_A = float(np.sum(eigvals))
        #else:
        #    print("Computing eigenvalues of A")
        #    if args.debug:
        #        t_eig_start = time.time()
        #    eigvals = np.linalg.eigvals(A)
        #    if args.debug:
        #        t_eig_end = time.time()
        #        print(f"[DEBUG] Eigvals computation time: {t_eig_end - t_eig_start:.4f}s")
        #    nuc_A = float(np.sum(eigvals))
        #    if args.save_eigvals:
        #        np.save(eigvals_file, eigvals)
        #        print(f"Saved eigenvalues to {eigvals_file}")
        nuc_A = np.trace(A) 
        print(f"Nuclear norm of A: {nuc_A:.6e}")
        results = {}
        for m in m_list:
            print(f"Running method={args.method} with m={m}")
            if args.debug:
                t_m_start = time.time()
            if args.method == 'colsample':
                A_nyst, t = nystrom_column_full(A, m=m, seed=args.seed, debug=args.debug)
            else:
                U_approx, S_approx, t = nystrom_rank_k_truncated(A, l=m, k=max(k_list), seed=args.seed, debug=args.debug)
                A_nyst = U_approx @ np.diag(S_approx) @ U_approx.T
                #A_nyst, t = nystrom_gaussian_full(A, m=m, seed=args.seed, debug=args.debug)
            if args.debug:
                t_nyst_end = time.time()
                print(f"[DEBUG] Nyström reconstruction time: {t_nyst_end - t_m_start:.4f}s")
            

            for k in k_list:
                # Slice factors for current k
                U_k = U_approx[:, :k]
                S_k = S_approx[:k]

                # Reconstruct the rank-k matrix (n x n)
                # Note: This matrix multiplication is expensive O(n^2 k), but necessary 
                # to compute the exact nuclear norm of the difference.
                A_approx_k = U_k @ (S_k[:, None] * U_k.T)

                if args.debug:
                    t_k_start = time.time()
                # Pass precomputed nuc_A to save time
                err = relative_nuclear_error(A, A_approx_k, nuc_A=nuc_A, debug=args.debug)

                print(f"k={k}: Relative Error = {err:.6e}")
                if args.debug:
                    t_k_end = time.time()
                    print(f"[DEBUG] Error comp for k={k} time: {t_k_end - t_k_start:.4f}s")
                results[(m, k)] = {
                    'time': float(t),
                    'nuc_A': float(nuc_A),
                    'rel_error': float(err),
                }
                print(f"m={m}, k={k}: time={t:.4f}s rel_error={err:.6e}")
            if args.debug:
                t_m_end = time.time()
                print(f"[DEBUG] Total for m={m}: {t_m_end - t_m_start:.4f}s")
        out_dir = args.out_dir
        # Append timestamp for uniqueness
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"results/{timestamp}"
        save_results(results, out_dir, meta={'dataset': 'MNIST', 'train': args.train, 'n': args.n, 'c': args.c, 'method': args.method})
        plot_results(results, out_dir, xlabel='k', title=f"rel_error n={args.n} c={args.c}")
        print(f"Saved results and plots to {out_dir}")
        if args.debug:
            t_sweep_end = time.time()
            print(f"[DEBUG] Total sweep time: {t_sweep_end - t_sweep_start:.4f}s")
        return

    # validate m/k for non-sweep modes
    if not args.sweep:
        if args.m is None or args.k is None:
            raise ValueError('For sequential or mpi mode you must provide --m and --k (unless --sweep is used)')

    if args.mode == 'sequential':
        run_sequential(args.n, args.c, args.m, args.k, args.method, args.seed, args.train, args.debug, args.approx_nuclear)
    else:
        run_mpi(args.n, args.c, args.m, args.k, args.seed, args.train, args.debug, args.approx_nuclear)


if __name__ == '__main__':
    main()
