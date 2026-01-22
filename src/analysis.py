def save_results(results: dict, out_dir: str, meta: dict = None) -> None:
    """Save sweep results to CSV and JSON metadata.

    `results` should be a dict keyed by (l,k) tuples with values containing
    at least: {'n': int, 'c': float, 'l': int, 'k': int, 'dataset': str,
               'sketch_method': str, 'compute_time': float, 'rel_error': float}
    """
    import os
    import json
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for (l, k), vals in sorted(results.items()):
        row = {}
        row.update(vals)
        rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, 'results.csv')
    df.to_csv(csv_path, index=False)

    if meta is not None:
        with open(os.path.join(out_dir, 'meta.json'), 'w') as jf:
            json.dump(meta, jf, indent=2)


def plot_results(results: dict, out_dir: str, xlabel: str = 'Approximation rank (k)', title: str = None) -> None:
    """Plot relative error and runtime from `results` and save PNGs in `out_dir`.

    Generates:
      - rel_error_vs_k.png
      - time_vs_k.png
      - parallel_time_vs_k.png (only if 'parallel_time' exists in results)
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict

    os.makedirs(out_dir, exist_ok=True)

    by_l = defaultdict(list)
    for (l, k), vals in sorted(results.items()):
        by_l[l].append((k, vals))

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('ggplot')

    sorted_ls = sorted(by_l.keys())
    cmap = plt.get_cmap('tab10')

    # PLOT 1: Relative Error vs k
    plt.figure(figsize=(10, 7))
    for idx, lval in enumerate(sorted_ls):
        items = sorted(by_l[lval], key=lambda x: x[0])
        ks = [it[0] for it in items]
        rels = [it[1].get('rel_error', None) for it in items]

        color = cmap(idx % 10)
        plt.plot(
            ks, rels, marker='o', linestyle='-', linewidth=2, markersize=6,
            label=f'Sketch size $l={lval}$', color=color, alpha=0.9
        )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(r'Relative Error ($\|A - \tilde{A}\|_* / \|A\|_*$)', fontsize=12)
    plt.yscale('log')

    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    else:
        if results:
            first_val = list(results.values())[0]
            n_val = first_val.get('n', '?')
            c_val = first_val.get('c', '?')
            plt.title(f'Nyström Error (n={n_val}, c={c_val})', fontsize=14, fontweight='bold')

    plt.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=10)
    plt.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.8)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rel_error_vs_k.png'), dpi=300)
    plt.close()

    # PLOT 2: Total runtime (compute_time) vs k
    plt.figure(figsize=(10, 7))
    for idx, lval in enumerate(sorted_ls):
        items = sorted(by_l[lval], key=lambda x: x[0])
        ks = [it[0] for it in items]
        times = [it[1].get('compute_time', 0.0) for it in items]

        color = cmap(idx % 10)
        plt.plot(
            ks, times, marker='s', linestyle='--', linewidth=2, markersize=6,
            label=f'Sketch size $l={lval}$', color=color, alpha=0.9
        )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Runtime (s)', fontsize=12)
    plt.title('Total Computation Time vs Rank', fontsize=14, fontweight='bold')
    plt.legend(frameon=True, fancybox=True, framealpha=0.9)
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'time_vs_k.png'), dpi=300)
    plt.close()

    # PLOT 3: Parallel kernel time vs k (only if present)
    has_parallel = any(('parallel_time' in v) for v in results.values())
    if has_parallel:
        plt.figure(figsize=(10, 7))
        for idx, lval in enumerate(sorted_ls):
            items = sorted(by_l[lval], key=lambda x: x[0])
            ks = [it[0] for it in items]
            ptimes = [it[1].get('parallel_time', 0.0) for it in items]

            color = cmap(idx % 10)
            plt.plot(
                ks, ptimes, marker='^', linestyle='-', linewidth=2, markersize=6,
                label=f'Sketch size $l={lval}$', color=color, alpha=0.9
            )

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel('Parallel kernel time (s)', fontsize=12)
        plt.title('Parallel Kernel Time vs Rank', fontsize=14, fontweight='bold')
        plt.legend(frameon=True, fancybox=True, framealpha=0.9)
        plt.grid(True, which='major', linestyle='-', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'parallel_time_vs_k.png'), dpi=300)
        plt.close()
