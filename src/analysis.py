def save_results(results: dict, out_dir: str, meta: dict = None) -> None:
    """Save sweep results to CSV and JSON metadata.

    `results` should be a dict keyed by (m,k) tuples with values containing
    at least: {'n': int, 'c': float, 'm': int, 'k': int, 'dataset': str, 
               'sketch_method': str, 'compute_time': float, 'rel_error': float}
    """
    import os
    import json
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for (m, k), vals in sorted(results.items()):
        row = {}
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

    