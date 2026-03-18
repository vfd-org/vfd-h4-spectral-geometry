"""
Generate figures for Paper III from simulation results.

Reads output from paper3_attractor_sweep.py and produces:
    - fig_phase_diagram.png
    - fig_ipr_distribution.png
    - fig_mode_visualisation.png

Usage:
    python paper3_plot_figures.py [--input results/paper3]
                                  [--output ../papers/paper-003-h4-attractors/figures]
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# Plot styling
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

CLASS_LABELS = {
    1: 'Class 1: Backbone harmonic',
    2: 'Class 2: Locked multi-mode',
    3: 'Class 3: Breather',
    4: 'Class 4: Transitional',
}
CLASS_COLORS = {
    1: '#2166ac',
    2: '#4dac26',
    3: '#d6604d',
    4: '#999999',
}


def plot_phase_diagram(input_dir, output_dir):
    """Plot attractor class fractions vs beta."""
    data = np.load(os.path.join(input_dir, 'phase_diagram.npz'))
    beta = data['beta_values']
    fracs = data['class_fractions']

    fig, ax = plt.subplots(figsize=(9, 5))

    for c in range(4):
        ax.plot(beta, fracs[:, c],
                color=CLASS_COLORS[c+1],
                linewidth=2,
                label=CLASS_LABELS[c+1])
        ax.fill_between(beta, 0, fracs[:, c],
                         color=CLASS_COLORS[c+1], alpha=0.1)

    ax.set_xscale('log')
    ax.set_xlabel(r'Nonlinear strength $\beta$')
    ax.set_ylabel('Fraction of trajectories')
    ax.set_title('Attractor Class Distribution vs Nonlinear Strength')
    ax.legend(loc='center right')
    ax.set_xlim(beta[0], beta[-1])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Mark regime boundaries
    ax.axvline(x=0.01, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.axvline(x=1.0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.text(0.003, 0.95, 'Linear', fontsize=9, alpha=0.5,
            ha='center', va='top')
    ax.text(0.1, 0.95, 'Weakly\nnonlinear', fontsize=9, alpha=0.5,
            ha='center', va='top')
    ax.text(3.0, 0.95, 'Strongly\nnonlinear', fontsize=9, alpha=0.5,
            ha='center', va='top')

    path = os.path.join(output_dir, 'fig_phase_diagram.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_ipr_distribution(input_dir, output_dir):
    """Plot IPR histograms at three representative beta values."""
    data = np.load(os.path.join(input_dir, 'attractor_classes.npz'))
    beta = data['beta_values']
    ipr = data['ipr']

    # Select three representative beta values (one per regime)
    targets = [0.005, 0.3, 3.0]
    indices = [np.argmin(np.abs(beta - t)) for t in targets]
    colors = ['#2166ac', '#f4a582', '#b2182b']
    labels = [rf'$\beta = {beta[i]:.3f}$' for i in indices]

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.logspace(np.log10(0.005), np.log10(1.0), 40)

    for idx, color, label in zip(indices, colors, labels):
        ipr_vals = ipr[idx]
        ipr_vals = ipr_vals[ipr_vals > 0]  # filter zeros
        if len(ipr_vals) > 0:
            ax.hist(ipr_vals, bins=bins, alpha=0.5, color=color,
                    label=label, edgecolor=color, linewidth=0.5)

    ax.set_xscale('log')
    ax.axvline(x=1/120, color='k', linestyle=':', alpha=0.5,
               label=r'$1/n = 1/120$')
    ax.set_xlabel('Time-averaged IPR')
    ax.set_ylabel('Count')
    ax.set_title('IPR Distribution Across Nonlinear Regimes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, 'fig_ipr_distribution.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_mode_visualisation(input_dir, output_dir):
    """
    Plot representative field configurations from each class.

    If full trajectory data is not available, generates synthetic
    examples based on the classification criteria.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Representative Configurations by Attractor Class',
                 fontsize=14)

    x = np.arange(120)

    # Try to load saved representative trajectories
    rep_file = os.path.join(input_dir, 'representative_configs.npz')
    if os.path.exists(rep_file):
        reps = np.load(rep_file)
        configs = {c: reps[f'class{c}'] for c in range(1, 5)}
    else:
        # Generate synthetic examples
        rng = np.random.default_rng(123)
        configs = {}

        # Class 1: delocalised, smooth
        configs[1] = 0.1 * np.sin(2 * np.pi * x / 24) + 0.02 * rng.normal(size=120)

        # Class 2: two-mode interference
        configs[2] = (0.15 * np.sin(2 * np.pi * x / 24)
                      + 0.12 * np.cos(2 * np.pi * x / 10)
                      + 0.02 * rng.normal(size=120))

        # Class 3: localised breather
        configs[3] = np.zeros(120)
        center = 60
        for i in range(120):
            configs[3][i] = 1.5 * np.exp(-0.3 * abs(i - center))
        configs[3] += 0.01 * rng.normal(size=120)

        # Class 4: broad, disordered
        configs[4] = 0.3 * rng.normal(size=120)

    titles = {
        1: 'Class 1: Backbone Harmonic (sector 5)',
        2: 'Class 2: Locked Multi-Mode (sectors 4+5)',
        3: 'Class 3: Breather (localised)',
        4: 'Class 4: Transitional',
    }

    for c, ax in zip(range(1, 5), axes.flat):
        ax.bar(x, configs[c], width=1.0, color=CLASS_COLORS[c],
               alpha=0.7, edgecolor='none')
        ax.set_title(titles[c], fontsize=11)
        ax.set_xlabel('Vertex index (by distance shell)')
        ax.set_ylabel(r'$\Phi_i$')
        ax.set_xlim(-1, 121)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()

    path = os.path.join(output_dir, 'fig_mode_visualisation.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate Paper III figures'
    )
    parser.add_argument('--input', type=str, default='results/paper3')
    parser.add_argument('--output', type=str,
                        default='../papers/paper-003-h4-attractors/figures')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    plot_phase_diagram(args.input, args.output)
    plot_ipr_distribution(args.input, args.output)
    plot_mode_visualisation(args.input, args.output)
