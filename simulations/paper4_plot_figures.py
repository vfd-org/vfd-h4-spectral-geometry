"""
Generate figures for Paper IV from simulation results.

Produces:
    - fig_mode_coupling_matrix.png (key figure)
    - fig_shell_occupancy.png
    - fig_stability_hierarchy.png
    - fig_coupling_control_comparison.png
    - fig_constraint_summary.png

Usage:
    python paper4_plot_figures.py [--input results/paper4]
                                  [--output ../papers/paper-004-h4-selection/figures]
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

CLASS_COLORS = {1: '#2166ac', 2: '#4dac26', 3: '#d6604d', 4: '#999999'}
CLASS_LABELS = {
    1: 'Class 1: Backbone', 2: 'Class 2: Locked',
    3: 'Class 3: Breather', 4: 'Class 4: Transitional',
}
SHELL_SIZES = [1, 12, 32, 42, 32, 1]  # 600-cell distance shells


def load_data(input_dir):
    """Load Paper 4 results."""
    data = np.load(os.path.join(input_dir, 'paper4_results.npz'),
                   allow_pickle=True)
    return data


def sector_label(eigenvalue, multiplicity):
    """Format sector label."""
    if abs(eigenvalue) < 0.01:
        return f"0\n(×{multiplicity})"
    return f"{eigenvalue:.1f}\n(×{multiplicity})"


def plot_coupling_matrix(data, output_dir):
    """
    Figure 1: 9x9 mode coupling constraint matrix.
    Main panel: heatmap. Right panel: sector participation bars.
    """
    M = data['h4_coupling_matrix']
    evals = data['sector_labels_evals']
    mults = data['sector_labels_mults']
    n = M.shape[0]

    # Labels
    labels = [f"S{i+1}" for i in range(n)]
    legend_text = [f"S{i+1}: λ={evals[i]:.2f} (×{int(mults[i])})"
                   for i in range(n)]

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)

    # Main heatmap
    ax_main = fig.add_subplot(gs[0])
    im = ax_main.imshow(M, cmap='magma', interpolation='nearest',
                        aspect='equal', vmin=0)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = M[i, j]
            if val > 0.02:
                color = 'white' if val > 0.3 else 'black'
                ax_main.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=7, color=color)
            elif val == 0 and i != j:
                ax_main.text(j, i, '×', ha='center', va='center',
                            fontsize=8, color='#cccccc', alpha=0.5)

    ax_main.set_xticks(range(n))
    ax_main.set_xticklabels(labels, fontsize=9)
    ax_main.set_yticks(range(n))
    ax_main.set_yticklabels(labels, fontsize=9)
    ax_main.set_xlabel('Spectral sector')
    ax_main.set_ylabel('Spectral sector')
    ax_main.set_title('Mode Coupling Constraint Matrix', fontsize=14)

    # Backbone box (S1-S6)
    rect = mpatches.Rectangle((-0.5, -0.5), 6, 6, linewidth=2,
                               edgecolor='#4dac26', facecolor='none',
                               linestyle='-', label='Backbone (S1–S6)')
    ax_main.add_patch(rect)

    # Conjugate pair boxes
    for (i, j) in [(1, 8), (2, 6)]:  # S2-S9, S3-S7
        rect = mpatches.Rectangle((j-0.4, i-0.4), 0.8, 0.8,
                                   linewidth=1.5, edgecolor='cyan',
                                   facecolor='none', linestyle='--')
        ax_main.add_patch(rect)
        rect2 = mpatches.Rectangle((i-0.4, j-0.4), 0.8, 0.8,
                                    linewidth=1.5, edgecolor='cyan',
                                    facecolor='none', linestyle='--')
        ax_main.add_patch(rect2)

    # Departure separator
    ax_main.axhline(y=5.5, color='white', linewidth=0.5, alpha=0.5)
    ax_main.axvline(x=5.5, color='white', linewidth=0.5, alpha=0.5)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
    cbar.set_label('Co-occurrence frequency')

    # Right panel: sector participation frequency
    ax_bar = fig.add_subplot(gs[1])
    participation = np.diag(M)
    colors = ['#2166ac'] * 6 + ['#d6604d'] * 3  # backbone vs departure
    ax_bar.barh(range(n), participation, color=colors, alpha=0.7,
                edgecolor='black', linewidth=0.5)
    ax_bar.set_yticks(range(n))
    ax_bar.set_yticklabels([])
    ax_bar.set_xlabel('Activation\nfrequency')
    ax_bar.set_ylim(-0.5, n - 0.5)
    ax_bar.invert_yaxis()
    ax_bar.grid(True, alpha=0.2, axis='x')

    # Legend for sector labels
    fig.text(0.02, 0.02, '  '.join(legend_text[:5]), fontsize=7,
             family='monospace', alpha=0.7)
    fig.text(0.02, -0.01, '  '.join(legend_text[5:]), fontsize=7,
             family='monospace', alpha=0.7)

    fig.tight_layout()
    path = os.path.join(output_dir, 'fig_mode_coupling_matrix.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def plot_shell_occupancy(data, output_dir):
    """Figure 2: Shell energy distribution per attractor class."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle('Spatial Shell Occupancy by Attractor Class', fontsize=14)

    shell_labels = [f'd={d}\n({SHELL_SIZES[d]}v)' for d in range(6)]
    x = np.arange(6)

    for cls, ax in zip([1, 2, 3], axes):
        means = data[f'shell_means_c{cls}']
        stds = data[f'shell_stds_c{cls}']

        ax.bar(x, means, yerr=stds, color=CLASS_COLORS[cls],
               alpha=0.7, edgecolor='black', linewidth=0.5,
               capsize=3, error_kw={'linewidth': 1})

        # Uniform reference
        uniform = np.array(SHELL_SIZES) / 120.0
        ax.plot(x, uniform, 'k--', alpha=0.3, linewidth=1,
                label='Uniform')

        ax.set_xticks(x)
        ax.set_xticklabels(shell_labels, fontsize=9)
        ax.set_ylabel('Energy fraction')
        ax.set_title(CLASS_LABELS[cls])
        ax.set_ylim(0, None)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    path = os.path.join(output_dir, 'fig_shell_occupancy.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_stability_hierarchy(data, output_dir):
    """Figure 3: Stability metrics per class."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle('Stability Hierarchy Across Attractor Classes', fontsize=14)

    classes = [1, 2, 3]
    colors = [CLASS_COLORS[c] for c in classes]
    labels = [CLASS_LABELS[c] for c in classes]

    # Panel 1: Count
    counts = [int(data[f'rankings_c{c}_count']) for c in classes]
    axes[0].bar(range(3), counts, color=colors, alpha=0.7,
                edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels([f'C{c}' for c in classes])
    axes[0].set_ylabel('Number of trajectories')
    axes[0].set_title('Attractor Counts')
    axes[0].grid(True, alpha=0.2, axis='y')

    # Panel 2: Persistence
    pers = [float(data[f'rankings_c{c}_persistence']) for c in classes]
    axes[1].bar(range(3), pers, color=colors, alpha=0.7,
                edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels([f'C{c}' for c in classes])
    axes[1].set_ylabel('Mean autocorrelation')
    axes[1].set_title('Temporal Persistence')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.2, axis='y')

    # Panel 3: IPR
    iprs = [float(data[f'rankings_c{c}_ipr']) for c in classes]
    axes[2].bar(range(3), iprs, color=colors, alpha=0.7,
                edgecolor='black', linewidth=0.5)
    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels([f'C{c}' for c in classes])
    axes[2].set_ylabel('Mean IPR')
    axes[2].set_title('Spatial Localisation')
    axes[2].axhline(y=1/120, color='k', linestyle=':', alpha=0.4,
                    label='1/n')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    path = os.path.join(output_dir, 'fig_stability_hierarchy.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_coupling_comparison(input_dir, output_dir):
    """Figure 4: H4 coupling matrix vs control."""
    data = load_data(input_dir)
    M_h4 = data['h4_coupling_matrix']
    n = M_h4.shape[0]

    # Load first control
    ctrl_path = os.path.join(input_dir, 'ctrl_coupling_0.npz')
    if not os.path.exists(ctrl_path):
        print(f"Skipping control comparison (no control data)")
        return

    ctrl = np.load(ctrl_path)
    M_ctrl = ctrl['M']
    n_ctrl = M_ctrl.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Mode Coupling: H$_4$ Graph vs Random Regular Graph',
                 fontsize=14)

    # H4
    vmax = max(np.max(M_h4), 0.01)
    im1 = axes[0].imshow(M_h4, cmap='magma', vmin=0, vmax=vmax,
                         interpolation='nearest', aspect='equal')
    axes[0].set_title(f'H$_4$ (600-cell)\n{n} sectors')
    axes[0].set_xlabel('Sector')
    axes[0].set_ylabel('Sector')
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels([f'S{i+1}' for i in range(n)], fontsize=8)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([f'S{i+1}' for i in range(n)], fontsize=8)
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Control (may have different number of sectors)
    n_show = min(n_ctrl, 20)  # show at most 20 sectors
    M_show = M_ctrl[:n_show, :n_show]
    im2 = axes[1].imshow(M_show, cmap='magma', vmin=0,
                         vmax=max(np.max(M_show), 0.01),
                         interpolation='nearest', aspect='equal')
    axes[1].set_title(f'Random 12-regular\n{n_ctrl} sectors (showing {n_show})')
    axes[1].set_xlabel('Sector')
    axes[1].set_ylabel('Sector')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    path = os.path.join(output_dir, 'fig_coupling_control_comparison.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_constraint_summary(data, output_dir):
    """Figure 5: Summary of empirical constraints."""
    M = data['h4_coupling_matrix']
    n = M.shape[0]
    sector_fracs = data['h4_sector_fracs']
    classes = data['h4_classes']

    # Compute constraint statistics
    n_pairs = n * (n - 1) // 2
    triu = np.triu_indices(n, k=1)
    pair_vals = M[triu]
    n_observed = np.sum(pair_vals > 0.02)
    n_forbidden = n_pairs - n_observed

    # Sector activation frequency
    diag = np.diag(M)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Empirical Selection Rules Summary', fontsize=14)

    # Left: allowed vs forbidden pairings
    ax = axes[0]
    ax.bar(['Observed\npairings', 'Absent\npairings'],
           [n_observed, n_forbidden],
           color=['#4dac26', '#d6604d'], alpha=0.7,
           edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Number of sector pairs')
    ax.set_title(f'Sector Coupling Rules\n({n_pairs} possible pairs)')
    ax.text(0, n_observed + 0.5, str(n_observed), ha='center',
            fontweight='bold')
    ax.text(1, n_forbidden + 0.5, str(n_forbidden), ha='center',
            fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

    # Right: backbone vs departure participation
    ax = axes[1]
    backbone_activation = np.mean(diag[:6])
    departure_activation = np.mean(diag[6:]) if n > 6 else 0
    ax.bar(['Backbone\n(S1–S6)', 'Departure\n(S7–S9)'],
           [backbone_activation, departure_activation],
           color=['#2166ac', '#d6604d'], alpha=0.7,
           edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Mean activation frequency')
    ax.set_title('Backbone vs Departure Participation')
    ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    path = os.path.join(output_dir, 'fig_constraint_summary.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paper IV figures')
    parser.add_argument('--input', type=str, default='results/paper4')
    parser.add_argument('--output', type=str,
                        default='../papers/paper-004-h4-selection/figures')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    data = load_data(args.input)

    plot_coupling_matrix(data, args.output)
    plot_shell_occupancy(data, args.output)
    plot_stability_hierarchy(data, args.output)
    plot_coupling_comparison(args.input, args.output)
    plot_constraint_summary(data, args.output)

    print("\nAll Paper IV figures generated.")
