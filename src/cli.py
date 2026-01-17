import argparse
import time
import datetime
from typing import Optional
import numpy as np
import os
from .dataset import load_mnist, rbf_kernel, load_year_prediction_msd
from .nystrom import nystrom_rank_k_truncated
from .analysis import save_results, plot_results
from .error import relative_nuclear_error
from .mpi_nystrom import nystrom_mpi


def run_sweep_sequential(n: int, c: float, l_list: list, k_list: list, sketch: str, dataset: str, dataset_path: str, seed: int, train: bool = True, debug: bool = False):
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
        U_approx, S_approx, t_compute = nystrom_rank_k_truncated(A, l=l, k=max(k_list), seed=seed, sketch=sketch, debug=debug)
        
        for k in k_list:
            # Slice factors for current k
            U_k = U_approx[:, :k]
            S_k = S_approx[:k]

            # Reconstruct the rank-k matrix (n x n)
            A_approx_k = U_k @ (S_k[:, None] * U_k.T)

            if debug:
                t_k_start = time.time()
            
            # Compute error
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


def run_sweep_mpi(n: int, c: float, l_list: list, k_list: list, sketch: str, dataset: str, dataset_path: str, seed: int, train: bool = True, debug: bool = False):
    from mpi4py import MPI
    from .mpi_nystrom import nystrom_mpi
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    q = int(np.sqrt(size))
    if q * q != size:
        if rank == 0: print(f"Error: Size {size} must be a perfect square.")
        comm.Abort(1)
    
    # Load and Build on Rank 0
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

    # Matrix A distribution
    if rank == 0:
        block_size = n // q
        for r in range(size):
            r_row, r_col = r // q, r % q
            
            # Use same indexing logic as the Send
            row_s, row_e = r_row * block_size, ((r_row + 1) * block_size if r_row < q - 1 else n)
            col_s, col_e = r_col * block_size, ((r_col + 1) * block_size if r_col < q - 1 else n)
            
            block = A_full[row_s:row_e, col_s:col_e]
            
            if r == 0:
                A_local = block
            else:
                # Use lowercase send (pickling) to handle shape variations
                comm.send(block, dest=r, tag=10)
    else:
        # Use lowercase recv
        A_local = comm.recv(source=0, tag=10)

    # Synchronized Sweep
    results = {}
    for l in l_list:
        if rank == 0:
            print(f"\n--- Running l={l} ---", flush=True)
        
        max_k = max(k_list)
        U_approx, S_approx, t_compute = nystrom_mpi(
            comm=comm, 
            A_local=A_local, 
            global_n=n, 
            k=max_k, 
            l=l, 
            seed=seed, 
            sketch=sketch, 
            debug=debug
        )
        
        if rank == 0:
            for k in k_list:
                U_k = U_approx[:, :k]
                S_k = S_approx[:k]
                A_approx_k = U_k @ (S_k[:, None] * U_k.T)
                err = relative_nuclear_error(A_full, A_approx_k, nuc_A=nuc_A)
                
                results[(l, k)] = {
                    'n': n, 'c': c, 'l': l, 'k': k,
                    'dataset': dataset, 'sketch_method': sketch,
                    'compute_time': t_compute, 'rel_error': err,
                    'memory_gb': float(A_full.nbytes / (1024**3)) if A_full is not None else 0.0
                }
                print(f"l={l}, k={k}: Time={t_compute:.4f}s Error={err:.6e}", flush=True)

        comm.Barrier()

    return results if rank == 0 else None


def main():
    parser = argparse.ArgumentParser(description='Randomized Nyström sweep experiments')
    parser.add_argument('--dataset', choices=['mnist', 'msd'], default='msd', help='dataset to use (mnist, msd), default: msd')
    parser.add_argument('--dataset-path', type=str, default='./datasets/YearPredictionMSD.txt', help='path to MSD dataset file')
    parser.add_argument('--n', type=int, required=True, help='number of data rows to use')
    parser.add_argument('--c', type=float, required=True, help='RBF bandwidth parameter')
    parser.add_argument('--l-list', type=str, required=True, help='comma-separated list of sketch sizes l')
    parser.add_argument('--k-list', type=str, required=True, help='comma-separated list of target ranks k')
    parser.add_argument('--sketch', choices=['gaussian', 'srht'], default='gaussian', help='sketching method (default: gaussian)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--train', action='store_true', help='use MNIST training set')
    parser.add_argument('--out-dir', type=str, default='results', help='output directory for results')
    parser.add_argument('--debug', action='store_true', help='enable debug prints with timing')
    parser.add_argument('--mpi', action='store_true', help='run with MPI for distributed computation')
    args = parser.parse_args()

    # Parse l and k lists
    l_list = [int(x) for x in args.l_list.split(',') if x.strip()]
    k_list = [int(x) for x in args.k_list.split(',') if x.strip()]
    
    if not l_list or not k_list:
        raise ValueError('--l-list and --k-list must contain at least one value')

    print(f"Sweep over l={l_list} and k={k_list}")

    # Sequantial / Parallel
    if args.mpi:
        results = run_sweep_mpi(args.n, args.c, l_list, k_list, args.sketch, args.dataset, args.dataset_path, args.seed, args.train, args.debug)
    else:
        results = run_sweep_sequential(args.n, args.c, l_list, k_list, args.sketch, args.dataset, args.dataset_path, args.seed, args.train, args.debug)

    # Save results, only on rank 0 for MPI
    if results is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"{args.out_dir}/{timestamp}"
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
            '#proc': os.getenv('OMPI_COMM_WORLD_SIZE', '1') if args.mpi else '1',
        })
        title = f"rel_error n={args.n} c={args.c} sketch={args.sketch}"
        plot_results(results, out_dir, xlabel='k')
        print(f"Saved results and plots to {out_dir}")


if __name__ == '__main__':
    main()

