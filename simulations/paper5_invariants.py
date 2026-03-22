"""
Paper V: Attractor Invariants and Scaling Relations.

Analyses invariant relationships and scaling laws within the attractor
space using data from the Paper 4 simulation ensemble.

No new GPU simulations required — operates on existing results.

Usage:
    python paper5_invariants.py [--input results/paper4]
                                [--output results/paper5]
                                [--figures ../papers/paper-005-h4-invariants-scaling/figures]
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

CLASS_COLORS = {1: '#2166ac', 2: '#4dac26', 3: '#d6604d'}
CLASS_LABELS = {1: 'Class 1: Backbone', 2: 'Class 2: Locked',
                3: 'Class 3: Breather'}


def load_data(input_dir):
    """Load Paper 4 results."""
    d = np.load(os.path.join(input_dir, 'paper4_results.npz'))
    return d


def compute_derived_quantities(sector_fracs, classes, ipr, persistence,
                               beta_values):
    """
    Compute derived invariant quantities for all stable attractors.

    Returns dict of arrays, one entry per stable attractor.
    """
    n_beta, n_ic, n_sectors = sector_fracs.shape

    # Flatten and filter stable (Classes 1-3)
    all_sf = []
    all_ipr = []
    all_pers = []
    all_cls = []
    all_beta = []

    for b in range(n_beta):
        for ic in range(n_ic):
            c = classes[b, ic]
            if c in [1, 2, 3]:
                all_sf.append(sector_fracs[b, ic])
                all_ipr.append(ipr[b, ic])
                all_pers.append(persistence[b, ic])
                all_cls.append(c)
                all_beta.append(beta_values[b])

    sf = np.array(all_sf)       # (N, 9)
    iprs = np.array(all_ipr)    # (N,)
    pers = np.array(all_pers)   # (N,)
    cls = np.array(all_cls)     # (N,)
    betas = np.array(all_beta)  # (N,)

    # Derived quantities
    backbone_frac = np.sum(sf[:, :6], axis=1)

    # Spectral width: number of active sectors (>10%)
    spectral_width = np.sum(sf > 0.10, axis=1).astype(float)

    # Spectral entropy: -sum(f * log(f))
    sf_safe = np.clip(sf, 1e-15, 1.0)
    spectral_entropy = -np.sum(sf_safe * np.log2(sf_safe), axis=1)

    # Top two ratio: f_max / f_second
    sorted_sf = np.sort(sf, axis=1)[:, ::-1]
    top1 = sorted_sf[:, 0]
    top2 = np.maximum(sorted_sf[:, 1], 1e-15)
    top_ratio = top1 / top2

    # Weighted spectral centroid
    sector_indices = np.arange(1, n_sectors + 1, dtype=float)
    spectral_centroid = np.sum(sf * sector_indices, axis=1)

    return {
        'sector_fracs': sf,
        'ipr': iprs,
        'persistence': pers,
        'cls': cls,
        'beta': betas,
        'backbone_frac': backbone_frac,
        'spectral_width': spectral_width,
        'spectral_entropy': spectral_entropy,
        'top_ratio': top_ratio,
        'spectral_centroid': spectral_centroid,
        'n_stable': len(iprs),
    }


def compute_scaling(data, beta_values):
    """Compute ensemble averages vs beta for scaling plots."""
    betas_unique = np.unique(data['beta'])
    scaling = {
        'beta': betas_unique,
        'mean_ipr': [], 'std_ipr': [],
        'mean_backbone': [], 'std_backbone': [],
        'mean_persistence': [], 'std_persistence': [],
        'mean_width': [], 'std_width': [],
        'frac_c1': [], 'frac_c2': [], 'frac_c3': [],
    }

    for b in betas_unique:
        mask = data['beta'] == b
        scaling['mean_ipr'].append(np.mean(data['ipr'][mask]))
        scaling['std_ipr'].append(np.std(data['ipr'][mask]))
        scaling['mean_backbone'].append(np.mean(data['backbone_frac'][mask]))
        scaling['std_backbone'].append(np.std(data['backbone_frac'][mask]))
        scaling['mean_persistence'].append(np.mean(data['persistence'][mask]))
        scaling['std_persistence'].append(np.std(data['persistence'][mask]))
        scaling['mean_width'].append(np.mean(data['spectral_width'][mask]))
        scaling['std_width'].append(np.std(data['spectral_width'][mask]))
        n = np.sum(mask)
        scaling['frac_c1'].append(np.sum(data['cls'][mask] == 1) / n)
        scaling['frac_c2'].append(np.sum(data['cls'][mask] == 2) / n)
        scaling['frac_c3'].append(np.sum(data['cls'][mask] == 3) / n)

    for k in scaling:
        if k != 'beta':
            scaling[k] = np.array(scaling[k])

    return scaling


def compute_correlations(data):
    """Compute key correlation statistics."""
    corrs = {}

    # IPR vs spectral entropy
    r, p = stats.spearmanr(data['spectral_entropy'], data['ipr'])
    corrs['ipr_vs_entropy'] = {'r': r, 'p': p}

    # Persistence vs backbone fraction
    r, p = stats.spearmanr(data['backbone_frac'], data['persistence'])
    corrs['persistence_vs_backbone'] = {'r': r, 'p': p}

    # IPR vs spectral width
    r, p = stats.spearmanr(data['spectral_width'], data['ipr'])
    corrs['ipr_vs_width'] = {'r': r, 'p': p}

    # IPR vs persistence
    r, p = stats.spearmanr(data['ipr'], data['persistence'])
    corrs['ipr_vs_persistence'] = {'r': r, 'p': p}

    return corrs


# ── Figure generation ─────────────────────────────────────

def plot_invariant_structure(data, output_dir):
    """
    Main figure: 3-panel invariant structure map.
    Panel A: IPR vs spectral entropy
    Panel B: Persistence vs backbone fraction
    Panel C: Backbone fraction distribution
    """
    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A — IPR vs Spectral Entropy
    ax_a = fig.add_subplot(gs[0, 0])
    for c in [1, 2, 3]:
        mask = data['cls'] == c
        ax_a.scatter(data['spectral_entropy'][mask], data['ipr'][mask],
                     c=CLASS_COLORS[c], s=12, alpha=0.5,
                     label=CLASS_LABELS[c], edgecolors='none')

    # Trend line
    sorted_idx = np.argsort(data['spectral_entropy'])
    window = max(len(sorted_idx) // 20, 5)
    ent_sorted = data['spectral_entropy'][sorted_idx]
    ipr_sorted = data['ipr'][sorted_idx]
    ipr_smooth = np.convolve(ipr_sorted, np.ones(window)/window, mode='valid')
    ent_smooth = ent_sorted[window//2:window//2+len(ipr_smooth)]
    ax_a.plot(ent_smooth, ipr_smooth, 'k-', linewidth=2, alpha=0.7,
              label='Moving average')

    ax_a.set_xlabel('Spectral entropy (bits)')
    ax_a.set_ylabel('IPR')
    ax_a.set_title('A. Localisation vs Spectral Complexity')
    ax_a.legend(fontsize=8, loc='upper left')
    ax_a.grid(True, alpha=0.2)

    # Panel B — Persistence vs Backbone Fraction
    ax_b = fig.add_subplot(gs[0, 1])
    for c in [1, 2, 3]:
        mask = data['cls'] == c
        ax_b.scatter(data['backbone_frac'][mask], data['persistence'][mask],
                     c=CLASS_COLORS[c], s=12, alpha=0.5,
                     label=CLASS_LABELS[c], edgecolors='none')

    # Linear fit
    slope, intercept, r, p, se = stats.linregress(
        data['backbone_frac'], data['persistence'])
    x_fit = np.linspace(data['backbone_frac'].min(),
                        data['backbone_frac'].max(), 100)
    ax_b.plot(x_fit, slope * x_fit + intercept, 'k-', linewidth=2,
              alpha=0.7, label=f'Linear fit (r={r:.2f})')

    ax_b.set_xlabel('Backbone fraction')
    ax_b.set_ylabel('Persistence ($C_{\\mathrm{max}}$)')
    ax_b.set_title('B. Stability vs Spectral Composition')
    ax_b.legend(fontsize=8, loc='lower right')
    ax_b.grid(True, alpha=0.2)

    # Panel C — Backbone Fraction Distribution (span bottom)
    ax_c = fig.add_subplot(gs[1, :])
    bins = np.linspace(0.4, 1.0, 40)
    for c in [1, 2, 3]:
        mask = data['cls'] == c
        ax_c.hist(data['backbone_frac'][mask], bins=bins,
                  alpha=0.5, color=CLASS_COLORS[c],
                  label=CLASS_LABELS[c], edgecolor=CLASS_COLORS[c],
                  linewidth=0.5)

    mean_bb = np.mean(data['backbone_frac'])
    std_bb = np.std(data['backbone_frac'])
    ax_c.axvline(mean_bb, color='k', linewidth=2, linestyle='-',
                 label=f'Mean = {mean_bb:.3f}')
    ax_c.axvspan(mean_bb - std_bb, mean_bb + std_bb, alpha=0.1,
                 color='gray', label=f'$\\pm 1\\sigma$ = {std_bb:.3f}')
    ax_c.set_xlabel('Backbone fraction')
    ax_c.set_ylabel('Count')
    ax_c.set_title('C. Distribution of Backbone Fraction Across Stable Attractors')
    ax_c.legend(fontsize=9)
    ax_c.grid(True, alpha=0.2)

    path = os.path.join(output_dir, 'fig_invariant_structure.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def plot_scaling(scaling, output_dir):
    """Scaling plots vs beta."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Scaling of Attractor Properties with Nonlinearity',
                 fontsize=14)

    beta = scaling['beta']

    # IPR vs beta
    ax = axes[0, 0]
    ax.errorbar(beta, scaling['mean_ipr'], yerr=scaling['std_ipr'],
                fmt='o-', color='#d6604d', markersize=5, capsize=3)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('Mean IPR')
    ax.set_title('Localisation')
    ax.axhline(1/120, color='k', linestyle=':', alpha=0.4, label='1/n')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Backbone fraction vs beta
    ax = axes[0, 1]
    ax.errorbar(beta, scaling['mean_backbone'], yerr=scaling['std_backbone'],
                fmt='s-', color='#2166ac', markersize=5, capsize=3)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('Mean backbone fraction')
    ax.set_title('Spectral Composition')
    ax.grid(True, alpha=0.2)

    # Persistence vs beta
    ax = axes[1, 0]
    ax.errorbar(beta, scaling['mean_persistence'],
                yerr=scaling['std_persistence'],
                fmt='D-', color='#4dac26', markersize=5, capsize=3)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('Mean persistence ($C_{\\mathrm{max}}$)')
    ax.set_title('Temporal Stability')
    ax.grid(True, alpha=0.2)

    # Spectral width vs beta
    ax = axes[1, 1]
    ax.errorbar(beta, scaling['mean_width'], yerr=scaling['std_width'],
                fmt='^-', color='#762a83', markersize=5, capsize=3)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('Mean spectral width')
    ax.set_title('Mode Participation')
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    path = os.path.join(output_dir, 'fig_scaling_relations.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_correlation_matrix(data, output_dir):
    """Correlation matrix between key quantities."""
    quantities = {
        'IPR': data['ipr'],
        'Persistence': data['persistence'],
        'Backbone\nfraction': data['backbone_frac'],
        'Spectral\nwidth': data['spectral_width'],
        'Spectral\nentropy': data['spectral_entropy'],
        'Top ratio': data['top_ratio'],
    }

    names = list(quantities.keys())
    n = len(names)
    corr = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            r, _ = stats.spearmanr(list(quantities.values())[i],
                                   list(quantities.values())[j])
            corr[i, j] = r

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1,
                   interpolation='nearest')

    for i in range(n):
        for j in range(n):
            color = 'white' if abs(corr[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                    fontsize=10, color=color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title('Spearman Correlation Between Attractor Quantities',
                 fontsize=13)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Spearman r')

    path = os.path.join(output_dir, 'fig_correlation_matrix.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def plot_geometry_spectrum(data, output_dir):
    """Geometry vs spectrum correspondence."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Geometry-Spectrum Correspondence', fontsize=14)

    # IPR vs spectral centroid
    ax = axes[0]
    for c in [1, 2, 3]:
        mask = data['cls'] == c
        ax.scatter(data['spectral_centroid'][mask], data['ipr'][mask],
                   c=CLASS_COLORS[c], s=12, alpha=0.5,
                   label=CLASS_LABELS[c], edgecolors='none')
    ax.set_xlabel('Spectral centroid (weighted sector index)')
    ax.set_ylabel('IPR')
    ax.set_title('Localisation vs Spectral Position')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Persistence vs IPR
    ax = axes[1]
    for c in [1, 2, 3]:
        mask = data['cls'] == c
        ax.scatter(data['ipr'][mask], data['persistence'][mask],
                   c=CLASS_COLORS[c], s=12, alpha=0.5,
                   label=CLASS_LABELS[c], edgecolors='none')
    ax.set_xlabel('IPR')
    ax.set_ylabel('Persistence ($C_{\\mathrm{max}}$)')
    ax.set_title('Stability vs Localisation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    path = os.path.join(output_dir, 'fig_geometry_spectrum.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='Paper V analysis')
    parser.add_argument('--input', type=str, default='results/paper4')
    parser.add_argument('--output', type=str, default='results/paper5')
    parser.add_argument('--figures', type=str,
                        default='../papers/paper-005-h4-invariants-scaling/figures')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)

    # Load data
    print("Loading Paper 4 data...")
    raw = load_data(args.input)

    # Compute derived quantities
    print("Computing derived quantities...")
    data = compute_derived_quantities(
        raw['h4_sector_fracs'], raw['h4_classes'],
        raw['h4_ipr'], raw['h4_persistence'],
        raw['beta_values'])
    print(f"  Stable attractors: {data['n_stable']}")

    # Scaling
    print("Computing scaling relations...")
    scaling = compute_scaling(data, raw['beta_values'])

    # Correlations
    print("Computing correlations...")
    corrs = compute_correlations(data)

    # Print key statistics
    print(f"\n=== PAPER 5 KEY STATISTICS ===")
    print(f"Stable attractors: {data['n_stable']}")
    print(f"Backbone fraction: {np.mean(data['backbone_frac']):.3f} "
          f"± {np.std(data['backbone_frac']):.3f}")
    print(f"Spectral width: {np.mean(data['spectral_width']):.2f} "
          f"± {np.std(data['spectral_width']):.2f}")
    print(f"Spectral entropy: {np.mean(data['spectral_entropy']):.3f} "
          f"± {np.std(data['spectral_entropy']):.3f}")
    print(f"IPR: {np.mean(data['ipr']):.4f} "
          f"± {np.std(data['ipr']):.4f}")
    print(f"Persistence: {np.mean(data['persistence']):.3f} "
          f"± {np.std(data['persistence']):.3f}")

    print(f"\nCorrelations (Spearman):")
    for name, c in corrs.items():
        print(f"  {name}: r={c['r']:.3f}, p={c['p']:.2e}")

    print(f"\nPer-class backbone fraction:")
    for c in [1, 2, 3]:
        mask = data['cls'] == c
        print(f"  Class {c}: {np.mean(data['backbone_frac'][mask]):.3f} "
              f"± {np.std(data['backbone_frac'][mask]):.3f}")

    # Save results
    np.savez(os.path.join(args.output, 'paper5_results.npz'),
             backbone_frac=data['backbone_frac'],
             spectral_width=data['spectral_width'],
             spectral_entropy=data['spectral_entropy'],
             ipr=data['ipr'],
             persistence=data['persistence'],
             cls=data['cls'],
             beta=data['beta'],
             top_ratio=data['top_ratio'],
             spectral_centroid=data['spectral_centroid'],
             scaling_beta=scaling['beta'],
             scaling_mean_ipr=scaling['mean_ipr'],
             scaling_mean_backbone=scaling['mean_backbone'],
             scaling_mean_persistence=scaling['mean_persistence'],
             )

    # Generate figures
    print("\n=== Generating figures ===")
    plot_invariant_structure(data, args.figures)
    plot_scaling(scaling, args.figures)
    plot_correlation_matrix(data, args.figures)
    plot_geometry_spectrum(data, args.figures)

    print(f"\nAll results saved to {args.output}/")
    print("All figures saved.")


if __name__ == '__main__':
    main()
