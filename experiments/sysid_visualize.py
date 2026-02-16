"""
System Identification Data Visualization

Generates plots to visualize the influence of control variables (batch_size, poll_interval)
on state variables (queue_length, cpu_util, mem_util, io_ops) from sysid experiment data.

Outputs:
- heatmap_state_vs_control.png: Heatmap of mean state per control combination
- scatter_control_vs_state.png: Scatter plots with linear regression
- correlation_matrix.png: Pearson correlation heatmap
- timeseries_step_response.png: Time-series step response
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

STATE_VARS = ['queue_length', 'cpu_util', 'mem_util', 'io_ops']
CONTROL_VARS = ['batch_size', 'poll_interval']
# Variables that accumulate over time — use delta per step instead of mean
DELTA_VARS = ['queue_length', 'mem_util']
# Variables that are instantaneous measurements — use mean per step
MEAN_VARS = ['cpu_util', 'io_ops']

STATE_LABELS = {
    'queue_length': 'Δ Queue Length (messages/step)',
    'cpu_util': 'CPU Utilization (%)',
    'mem_util': 'Δ Memory Utilization (%/step)',
    'io_ops': 'I/O Operations (bytes/s)',
}
CONTROL_LABELS = {
    'batch_size': 'Batch Size',
    'poll_interval': 'Poll Interval (ms)',
}


def load_and_prepare(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['elapsed_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    # Filter only main phase data (exclude settle phases)
    df = df[~df['phase'].str.contains('settle', na=False)].reset_index(drop=True)
    print(f"Loaded {len(df)} samples, {df[['batch_size','poll_interval']].drop_duplicates().shape[0]} unique control combinations")
    return df


def compute_step_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-step aggregated metrics that remove temporal accumulation bias.

    For accumulating variables (queue_length, mem_util): use delta (end - start) per step.
    For instantaneous variables (cpu_util, io_ops): use mean per step.

    Returns DataFrame with 1 row per unique (batch_size, poll_interval) combination.
    """
    # Identify step boundaries by detecting changes in control variables
    ctrl_shifted = df[CONTROL_VARS].shift(1)
    step_boundary = (df[CONTROL_VARS] != ctrl_shifted).any(axis=1)
    df = df.copy()
    df['step_id'] = step_boundary.cumsum()

    rows = []
    for step_id, group in df.groupby('step_id'):
        row = {
            'batch_size': group['batch_size'].iloc[0],
            'poll_interval': group['poll_interval'].iloc[0],
        }
        # Delta for accumulating variables
        for var in DELTA_VARS:
            row[var] = group[var].iloc[-1] - group[var].iloc[0]
        # Mean for instantaneous variables
        for var in MEAN_VARS:
            row[var] = group[var].mean()
        rows.append(row)

    df_steps = pd.DataFrame(rows)
    print(f"Computed step deltas: {len(df_steps)} steps")
    return df_steps


def plot_heatmaps(df_steps: pd.DataFrame, output_dir: str):
    """
    Heatmap of state variable per (batch_size, poll_interval) combination.
    Uses delta for accumulating vars, mean for instantaneous vars.
    Dynamically sizes figure based on grid dimensions.
    """
    # Detect grid size
    n_batch = df_steps['batch_size'].nunique()
    n_poll = df_steps['poll_interval'].nunique()

    # Dynamic figure size: scale with grid size, minimum 14×11
    width = max(14, n_batch * 0.7)
    height = max(11, n_poll * 0.6)

    fig, axes = plt.subplots(2, 2, figsize=(width, height))
    fig.suptitle('State Variables per Control Combination (delta for queue/mem, mean for cpu/io)',
                 fontsize=12, fontweight='bold')

    # Disable annotations for very large grids (reduces clutter)
    annot = True
    fmt = '.1f'
    annot_fontsize = 8

    if n_batch * n_poll > 100:
        annot = False
        print(f"Large grid detected ({n_batch}×{n_poll}): disabling heatmap annotations")
    elif n_batch * n_poll > 50:
        annot_fontsize = 6  # Smaller font for medium-large grids

    for idx, state_var in enumerate(STATE_VARS):
        ax = axes[idx // 2][idx % 2]

        pivot_table = df_steps.pivot_table(
            index='poll_interval', columns='batch_size', values=state_var
        )
        pivot_table = pivot_table.sort_index(ascending=False)

        sns.heatmap(
            pivot_table, ax=ax, annot=annot, fmt=fmt, cmap='YlOrRd',
            linewidths=0.5, cbar_kws={'label': STATE_LABELS[state_var]},
            annot_kws={'fontsize': annot_fontsize} if annot else {}
        )
        ax.set_title(STATE_LABELS[state_var], fontsize=11)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Poll Interval (ms)')

    plt.tight_layout()
    path = os.path.join(output_dir, 'heatmap_state_vs_control.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap for {n_batch}×{n_poll} grid: {path}")


def plot_scatter_regression(df_steps: pd.DataFrame, output_dir: str):
    """Scatter plots of each control variable vs each state variable with regression line.
    Uses step-aggregated data (delta for accumulating vars, mean for instantaneous)."""
    fig, axes = plt.subplots(len(STATE_VARS), len(CONTROL_VARS), figsize=(14, 16))
    fig.suptitle('Control vs State Variable Relationships (with Linear Regression)', fontsize=14, fontweight='bold')

    for i, state_var in enumerate(STATE_VARS):
        for j, ctrl_var in enumerate(CONTROL_VARS):
            ax = axes[i][j]
            x = df_steps[ctrl_var].values
            y = df_steps[state_var].values

            ax.scatter(x, y, alpha=0.7, s=60, edgecolors='white', linewidth=0.5)

            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'r={r_value:.3f}, p={p_value:.3f}')

            ax.set_xlabel(CONTROL_LABELS[ctrl_var])
            ax.set_ylabel(STATE_LABELS[state_var])
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'scatter_control_vs_state.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_scatter_separated(df_steps: pd.DataFrame, output_dir: str):
    """Separated scatter plots: one subplot per value of the other control variable.

    Generates 2 PNGs:
    - scatter_by_batch_size.png: X=batch_size, columns=poll_interval values (4 rows x 5 cols)
    - scatter_by_poll_interval.png: X=poll_interval, columns=batch_size values (4 rows x 5 cols)
    """
    for ctrl_var, other_var in [('batch_size', 'poll_interval'), ('poll_interval', 'batch_size')]:
        other_values = sorted(df_steps[other_var].unique())
        n_cols = len(other_values)

        fig, axes = plt.subplots(len(STATE_VARS), n_cols, figsize=(4 * n_cols, 3.5 * len(STATE_VARS)))
        fig.suptitle(f'{CONTROL_LABELS[ctrl_var]} vs State — separated by {CONTROL_LABELS[other_var]}',
                     fontsize=14, fontweight='bold')

        for i, state_var in enumerate(STATE_VARS):
            for j, other_val in enumerate(other_values):
                ax = axes[i][j]
                subset = df_steps[df_steps[other_var] == other_val]
                x = subset[ctrl_var].values
                y = subset[state_var].values

                ax.plot(x, y, 'o-', markersize=7, linewidth=1.5)

                # Regression line + r value
                if len(x) >= 3:
                    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
                    x_line = np.linspace(x.min(), x.max(), 50)
                    ax.plot(x_line, slope * x_line + intercept, 'r--', alpha=0.5, linewidth=1)
                    ax.set_title(f'{CONTROL_LABELS[other_var]}={int(other_val)}  r={r_value:.2f}', fontsize=9)
                else:
                    ax.set_title(f'{CONTROL_LABELS[other_var]}={int(other_val)}', fontsize=9)

                if j == 0:
                    ax.set_ylabel(STATE_LABELS[state_var], fontsize=8)
                if i == len(STATE_VARS) - 1:
                    ax.set_xlabel(CONTROL_LABELS[ctrl_var], fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, f'scatter_by_{ctrl_var}.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")


def plot_correlation_matrix(df_steps: pd.DataFrame, output_dir: str):
    """Pearson correlation heatmap using step-aggregated data (bias-free)."""
    all_vars = CONTROL_VARS + STATE_VARS
    labels = [CONTROL_LABELS.get(v, STATE_LABELS.get(v, v)) for v in all_vars]

    corr = df_steps[all_vars].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('Pearson Correlation Matrix (Control + State Variables)', fontsize=14, fontweight='bold')

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, ax=ax, annot=True, fmt='.3f', cmap='RdBu_r',
        vmin=-1, vmax=1, center=0, linewidths=0.5,
        xticklabels=labels, yticklabels=labels,
        mask=mask,
    )
    ax.set_xticklabels(labels, rotation=45, ha='right')

    plt.tight_layout()
    path = os.path.join(output_dir, 'correlation_matrix.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_timeseries(df: pd.DataFrame, output_dir: str):
    """Time-series of state variables with control step indicators."""
    fig, axes = plt.subplots(len(STATE_VARS) + 1, 1, figsize=(16, 14), sharex=True)
    fig.suptitle('Step Response: State Variables Over Time', fontsize=14, fontweight='bold')

    elapsed = df['elapsed_sec'].values

    # Plot control variables on top subplot
    ax_ctrl = axes[0]
    ax_ctrl.step(elapsed, df['batch_size'].values, where='post', color='tab:blue', label='Batch Size', linewidth=1.5)
    ax_ctrl2 = ax_ctrl.twinx()
    ax_ctrl2.step(elapsed, df['poll_interval'].values, where='post', color='tab:orange', label='Poll Interval (ms)', linewidth=1.5)
    ax_ctrl.set_ylabel('Batch Size', color='tab:blue')
    ax_ctrl2.set_ylabel('Poll Interval (ms)', color='tab:orange')
    ax_ctrl.set_title('Control Variables')
    lines1, labels1 = ax_ctrl.get_legend_handles_labels()
    lines2, labels2 = ax_ctrl2.get_legend_handles_labels()
    ax_ctrl.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    ax_ctrl.grid(True, alpha=0.3)

    # Add vertical lines at step boundaries
    step_changes = df[['batch_size', 'poll_interval']].diff().fillna(0)
    change_idx = step_changes[(step_changes['batch_size'] != 0) | (step_changes['poll_interval'] != 0)].index

    colors = plt.cm.tab20(np.linspace(0, 1, 25))

    for idx, state_var in enumerate(STATE_VARS):
        ax = axes[idx + 1]
        ax.plot(elapsed, df[state_var].values, linewidth=0.8, alpha=0.8)

        # Mark step boundaries
        for ci in change_idx:
            ax.axvline(x=elapsed[ci], color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

        ax.set_ylabel(STATE_LABELS[state_var], fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Elapsed Time (seconds)')
    plt.tight_layout()
    path = os.path.join(output_dir, 'timeseries_step_response.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize System Identification Data')
    parser.add_argument('data_file', help='Path to experiment data CSV')
    parser.add_argument('--output-dir', default=None, help='Output directory (default: same as data file)')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.data_file) or './results'
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_and_prepare(args.data_file)

    # Compute per-step aggregates (delta for accumulating vars, mean for instantaneous)
    df_steps = compute_step_deltas(df)

    print("\n--- Generating heatmaps ---")
    plot_heatmaps(df_steps, args.output_dir)

    print("\n--- Generating scatter plots ---")
    plot_scatter_regression(df_steps, args.output_dir)

    print("\n--- Generating separated scatter plots ---")
    plot_scatter_separated(df_steps, args.output_dir)

    print("\n--- Generating correlation matrix ---")
    plot_correlation_matrix(df_steps, args.output_dir)

    print("\n--- Generating time-series (raw data) ---")
    plot_timeseries(df, args.output_dir)

    print(f"\nAll visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
