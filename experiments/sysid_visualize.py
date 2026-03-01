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

STATE_VARS = ['queue_length', 'queue_length_mean', 'cpu_util', 'mem_util', 'indexing_time_rate', 'io_write_ops', 'os_cpu_percent', 'os_mem_used_percent', 'gc_time_rate', 'write_queue_size']
CONTROL_VARS = ['batch_size', 'poll_interval']
# Variables that accumulate over time — use delta per step instead of mean
DELTA_VARS = ['queue_length']
# Variables that are instantaneous measurements — use mean per step
MEAN_VARS = ['cpu_util', 'indexing_time_rate', 'io_write_ops', 'os_cpu_percent', 'os_mem_used_percent', 'gc_time_rate', 'write_queue_size']
# Variables where max per step is more meaningful (e.g. JVM heap peak before GC)
MAX_VARS = ['mem_util']

STATE_LABELS = {
    'queue_length': 'Δ Queue Length (messages/step)',
    'queue_length_mean': 'Queue Length Mean (messages)',
    'cpu_util': 'ES Process CPU (%)',
    'mem_util': 'ES JVM Heap Used (%)',
    'indexing_time_rate': 'Indexing Time Rate (ms/s)',
    'io_write_ops': 'Disk I/O Write Ops/s',
    'os_cpu_percent': 'OS CPU (%)',
    'os_mem_used_percent': 'OS RAM Used (%)',
    'gc_time_rate': 'GC Time Rate (ms/s)',
    'write_queue_size': 'Write Thread Pool Queue',
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
            row[f'{var}_mean'] = group[var].mean()
        # Mean for instantaneous variables
        for var in MEAN_VARS:
            row[var] = group[var].mean()
        # Max for GC-affected variables (peak before GC)
        for var in MAX_VARS:
            row[var] = group[var].max()
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

    n_vars = len(STATE_VARS)
    n_cols = 2
    n_rows = (n_vars + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height * n_rows / 2))
    fig.suptitle('State Variables per Control Combination (delta for queue, mean for others)',
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
        ax = axes[idx // n_cols][idx % n_cols]

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
    # Only plot vars that exist in raw data (exclude computed vars like queue_length_mean)
    raw_vars = [v for v in STATE_VARS if v in df.columns]
    fig, axes = plt.subplots(len(raw_vars) + 1, 1, figsize=(16, 14), sharex=True)
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

    for idx, state_var in enumerate(raw_vars):
        ax = axes[idx + 1]
        ax.plot(elapsed, df[state_var].values, linewidth=0.8, alpha=0.8)

        # Mark step boundaries
        for ci in change_idx:
            ax.axvline(x=elapsed[ci], color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

        ax.set_ylabel(STATE_LABELS.get(state_var, state_var), fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Elapsed Time (seconds)')
    plt.tight_layout()
    path = os.path.join(output_dir, 'timeseries_step_response.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


STALL_THRESHOLD = 319  # batch_size above which system stall may occur


def plot_stall_detection(df_steps: pd.DataFrame, output_dir: str):
    """Deteksi stall (batch_size > 319) dan outlier (IQR) pada data eksperimen."""
    KEY_VARS = ['cpu_util', 'io_write_ops', 'queue_length']

    df_steps = df_steps.copy()
    df_steps['is_stall'] = df_steps['batch_size'] > STALL_THRESHOLD

    normal = df_steps[~df_steps['is_stall']]
    stall = df_steps[df_steps['is_stall']]

    # IQR outlier detection
    outlier_flags = {}
    outlier_stats = {}
    for var in STATE_VARS:
        Q1 = df_steps[var].quantile(0.25)
        Q3 = df_steps[var].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        flag = (df_steps[var] < lower) | (df_steps[var] > upper)
        outlier_flags[var] = flag
        outlier_stats[var] = {'count': int(flag.sum()), 'pct': flag.mean() * 100}

    # --- Plot 2x2 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Deteksi Stall & Outlier', fontsize=14, fontweight='bold')

    for idx, var in enumerate(KEY_VARS):
        ax = axes[idx // 2][idx % 2]
        norm_sub = df_steps[~df_steps['is_stall']]
        stall_sub = df_steps[df_steps['is_stall']]
        outliers = df_steps[outlier_flags[var]]

        ax.scatter(norm_sub['batch_size'], norm_sub[var], c='steelblue',
                   alpha=0.6, s=40, label='Normal', zorder=2)
        ax.scatter(stall_sub['batch_size'], stall_sub[var], c='crimson',
                   alpha=0.6, s=40, label=f'Stall (bs>{STALL_THRESHOLD})', zorder=2)
        ax.scatter(outliers['batch_size'], outliers[var],
                   facecolors='none', edgecolors='orange', s=80,
                   linewidths=1.5, label='Outlier (IQR)', zorder=3)
        ax.axvline(x=STALL_THRESHOLD, color='gray', linestyle='--',
                   alpha=0.7, label=f'Threshold={STALL_THRESHOLD}')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel(STATE_LABELS.get(var, var))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Summary text panel
    ax_sum = axes[1][1]
    ax_sum.axis('off')
    lines = [f"Total steps: {len(df_steps)}",
             f"Normal: {len(normal)}  |  Stall (bs>{STALL_THRESHOLD}): {len(stall)}",
             "",
             "Perbandingan rata-rata (normal vs stall):"]
    for var in KEY_VARS:
        mn = normal[var].mean() if len(normal) > 0 else 0
        ms = stall[var].mean() if len(stall) > 0 else 0
        lines.append(f"  {var:25s}: {mn:10.1f} vs {ms:10.1f}")
    lines.append("")
    lines.append("Outlier per variabel (IQR):")
    for var in STATE_VARS:
        s = outlier_stats[var]
        lines.append(f"  {var:25s}: {s['count']:3d} ({s['pct']:.1f}%)")
    ax_sum.text(0.05, 0.95, '\n'.join(lines), transform=ax_sum.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(output_dir, 'stall_outlier_detection.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # Console output
    print(f"\n  Threshold stall: batch_size > {STALL_THRESHOLD}")
    print(f"  Normal: {len(normal)} steps | Stall: {len(stall)} steps")
    print(f"\n  Perbandingan rata-rata (normal vs stall):")
    for var in KEY_VARS:
        mn = normal[var].mean() if len(normal) > 0 else 0
        ms = stall[var].mean() if len(stall) > 0 else 0
        print(f"    {var:25s}: {mn:10.1f} vs {ms:10.1f}")
    print(f"\n  Outlier per variabel (IQR):")
    for var in STATE_VARS:
        s = outlier_stats[var]
        print(f"    {var:25s}: {s['count']:3d} ({s['pct']:.1f}%)")


def plot_nonlinearity_check(df_steps: pd.DataFrame, output_dir: str):
    """Analisis non-linearitas: regresi polinomial derajat 1/2/3 untuk batch_size -> state."""
    x = df_steps['batch_size'].values
    results = {}

    for var in STATE_VARS:
        y = df_steps[var].values
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = {}
        coeffs_all = {}
        for degree in [1, 2, 3]:
            coeffs = np.polyfit(x, y, degree)
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)
            r2[degree] = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            coeffs_all[degree] = coeffs
        results[var] = {'r2': r2, 'coeffs': coeffs_all}

    # --- Figure 1: scatter + polynomial fits ---
    n_vars = len(STATE_VARS)
    n_cols = 2
    n_rows = (n_vars + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
    fig.suptitle('Analisis Non-linearitas: Regresi Polinomial (batch_size -> state)',
                 fontsize=13, fontweight='bold')

    x_smooth = np.linspace(x.min(), x.max(), 200)
    fit_styles = {1: ('steelblue', '--', 'Linear'),
                  2: ('darkorange', '-', 'Kuadratik'),
                  3: ('green', '-.', 'Kubik')}

    for idx, var in enumerate(STATE_VARS):
        ax = axes[idx // n_cols][idx % n_cols]
        y = df_steps[var].values
        ax.scatter(x, y, alpha=0.4, s=25, c='gray', zorder=1)

        for degree in [1, 2, 3]:
            coeffs = results[var]['coeffs'][degree]
            y_fit = np.polyval(coeffs, x_smooth)
            color, ls, name = fit_styles[degree]
            r2_val = results[var]['r2'][degree]
            ax.plot(x_smooth, y_fit, color=color, linestyle=ls,
                    linewidth=2, label=f'{name}: R²={r2_val:.4f}', zorder=2)

        ax.set_xlabel('Batch Size')
        ax.set_ylabel(STATE_LABELS.get(var, var))
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

    if n_vars % 2 == 1:
        axes[-1][-1].axis('off')

    plt.tight_layout()
    path = os.path.join(output_dir, 'nonlinearity_polyfit.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Figure 2: R² comparison bar chart ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.suptitle('Perbandingan R² per Derajat Polinomial (batch_size -> state)',
                  fontsize=13, fontweight='bold')

    var_names = [STATE_LABELS.get(v, v) for v in STATE_VARS]
    y_pos = np.arange(len(STATE_VARS))
    bar_height = 0.25

    for i, degree in enumerate([1, 2, 3]):
        color, _, name = fit_styles[degree]
        r2_vals = [results[v]['r2'][degree] for v in STATE_VARS]
        ax2.barh(y_pos + i * bar_height, r2_vals, bar_height,
                 label=name, color=color, alpha=0.8)

    ax2.set_yticks(y_pos + bar_height)
    ax2.set_yticklabels(var_names, fontsize=9)
    ax2.set_xlabel('R²')
    ax2.legend(fontsize=10)
    ax2.grid(True, axis='x', alpha=0.3)
    ax2.invert_yaxis()

    plt.tight_layout()
    path2 = os.path.join(output_dir, 'nonlinearity_r2_comparison.png')
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    print(f"Saved: {path2}")

    # Console output
    print(f"\n  {'Variabel':30s} | {'Linear R²':>10s} | {'Quad R²':>10s} | {'Cubic R²':>10s} | {'Gain (Q-L)':>10s}")
    print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for var in STATE_VARS:
        r = results[var]['r2']
        gain = r[2] - r[1]
        marker = ' ***' if gain > 0.05 else ''
        print(f"  {var:30s} | {r[1]:10.4f} | {r[2]:10.4f} | {r[3]:10.4f} | {gain:+10.4f}{marker}")
    print(f"\n  *** = non-linearitas signifikan (gain > 0.05)")


def main():
    parser = argparse.ArgumentParser(description='Visualize System Identification Data')
    parser.add_argument('data_file', help='Path to experiment data CSV')
    parser.add_argument('--output-dir', default=None, help='Output directory (default: same as data file)')
    parser.add_argument('--batch-max', type=int, default=None, help='Filter: max batch_size to include')
    parser.add_argument('--poll-max', type=int, default=None, help='Filter: max poll_interval to include')
    args = parser.parse_args()

    if args.output_dir is None:
        # Auto-detect: if CSV is inside a runs/run_*/results/ directory, output to viz/
        from pathlib import Path
        csv_path = Path(args.data_file).resolve()
        if 'runs' in csv_path.parts and csv_path.parent.name == 'results':
            args.output_dir = str(csv_path.parent.parent / 'viz')
        else:
            args.output_dir = os.path.dirname(args.data_file) or './results'
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_and_prepare(args.data_file)

    # Apply filters if specified
    if args.batch_max is not None:
        before = len(df)
        df = df[df['batch_size'] <= args.batch_max].reset_index(drop=True)
        print(f"Filter batch_size <= {args.batch_max}: {before} -> {len(df)} samples")
    if args.poll_max is not None:
        before = len(df)
        df = df[df['poll_interval'] <= args.poll_max].reset_index(drop=True)
        print(f"Filter poll_interval <= {args.poll_max}: {before} -> {len(df)} samples")

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

    print("\n--- Deteksi stall & outlier ---")
    plot_stall_detection(df_steps, args.output_dir)

    print("\n--- Analisis non-linearitas ---")
    plot_nonlinearity_check(df_steps, args.output_dir)

    print(f"\nAll visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
