"""
Closed-Loop Experiment Visualization — Grouped by Q Configuration

Generates thesis-ready plots comparing 5 controllers per Q config:
  Static, Rule-Based, PID, LQR(K_Q), ANN(model_Q)

For each Q config (Q1–Q4), LQR and ANN are resolved to the matched variant
(e.g., Q1 → lqr_q1, ann_q1). Static/Rule-Based/PID are the same across all Q.

Outputs per Q config (×3 = Q1, Q2, Q4):
  1. timeseries_queue_grid_Q{n}.png   — Queue length time-series (4×1 grid by pattern)
  2. timeseries_batch_grid_Q{n}.png   — Batch size time-series
  3. bar_backlog_mean_Q{n}.png        — Backlog mean bar chart per pattern
  4. bar_backlog_max_Q{n}.png         — Backlog max bar chart per pattern
  5. bar_throughput_Q{n}.png          — Throughput bar chart per pattern
  6. bar_efficiency_Q{n}.png          — CPU & memory efficiency per pattern
  7. heatmap_summary_Q{n}.png         — Heatmap: controller × pattern for key metrics
  8. injection_vs_queue_Q{n}.png      — Injection rate overlay with queue length
  9. boxplot_queue_Q{n}.png           — Queue length distribution boxplots
  10. radar_comparison_Q{n}.png       — Radar/spider chart per pattern

Plus aggregate cost/regret plots:
  11. bar_cost_J.png                  — Cost J per Q (5 controllers per subplot)
  12. heatmap_cost_J.png              — Heatmap: 5 controllers × 4 Q configs
  13. bar_regret.png                  — Normalized regret (5 controllers)
  14. bar_cost_by_pattern.png         — Cost J breakdown per pattern per Q

Usage:
    python closed_loop_visualize.py --csv <path_to_closed_loop_data.csv> [--output-dir <dir>]
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.lines import Line2D

from metrics_analysis import (
    analyze_experiment, Q_CONFIGS, R_MATRIX, STATE_VARS, X_TARGET, CONTROL_VARS,
    Q_MATCHED_CONTROLLERS, BASELINE_CONTROLLERS, GENERIC_CONTROLLER_ORDER,
)

# ─── Style ────────────────────────────────────────────────────────────

# 5 generic controllers (LQR/ANN resolved per Q config)
CONTROLLER_COLORS = {
    'static':     '#888888',
    'rule_based': '#2196F3',
    'pid':        '#FF9800',
    'lqr':        '#4CAF50',
    'ann':        '#E91E63',
    'passive':    '#000000',
}

CONTROLLER_LABELS = {
    'static':     'Static',
    'rule_based': 'Rule-Based',
    'pid':        'PID',
    'lqr':        'LQR',
    'ann':        'ANN (MLE)',
    'passive':    'Passive (No Drain)',
}

PATTERN_LABELS = {
    'step':          'Step (Constant)',
    'ramp':          'Ramp (Linear)',
    'impulse':       'Impulse (Burst)',
    'periodic_step': 'Periodic Step',
    'step_low':      'Step Low (100 msg/s)',
}

CONTROLLER_ORDER = GENERIC_CONTROLLER_ORDER  # ['static', 'rule_based', 'pid', 'lqr', 'ann']
PATTERN_ORDER = ['step', 'ramp', 'impulse', 'periodic_step', 'step_low']

sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
})


def resolve_controller(generic_mode, q_name):
    """Map generic 'lqr'/'ann' to Q-specific variant for data lookup."""
    if generic_mode in ('lqr', 'ann'):
        return Q_MATCHED_CONTROLLERS[q_name][generic_mode]
    return generic_mode  # static, rule_based, pid unchanged


def q_short(q_name):
    """Extract short Q label, e.g. 'Q1_backlog' -> 'Q1'."""
    return q_name.split('_')[0].upper()


def _controller_legend(ax=None):
    """Add a shared controller legend."""
    handles = [
        Line2D([0], [0], color=CONTROLLER_COLORS[m], lw=2, label=CONTROLLER_LABELS[m])
        for m in CONTROLLER_ORDER
    ]
    target = ax or plt
    target.legend(handles=handles, loc='upper right', framealpha=0.9)


# ─── 1. Time-series Grid: Queue Length ────────────────────────────────

def plot_timeseries_queue_grid(df, output_dir, q_name):
    patterns = [p for p in PATTERN_ORDER if p in df['load_pattern'].unique()]
    fig, axes = plt.subplots(len(patterns), 1, figsize=(14, 3.5 * len(patterns)),
                             sharex=False)
    if len(patterns) == 1:
        axes = [axes]

    for ax, pattern in zip(axes, patterns):
        for mode in CONTROLLER_ORDER:
            actual = resolve_controller(mode, q_name)
            sub = df[(df['controller_mode'] == actual) & (df['load_pattern'] == pattern)]
            sub = sub.sort_values('elapsed_sec')
            if len(sub) == 0:
                continue
            ax.plot(sub['elapsed_sec'], sub['queue_length'],
                    color=CONTROLLER_COLORS[mode], alpha=0.85, lw=1.2,
                    label=CONTROLLER_LABELS[mode])

        ax.set_title(PATTERN_LABELS[pattern], fontweight='bold')
        ax.set_ylabel('Queue Length')
        ax.set_xlabel('Time (s)')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    _controller_legend(axes[0])
    qs = q_short(q_name)
    fig.suptitle(f'Queue Length Over Time by Load Pattern ({qs})', fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, f'timeseries_queue_grid_{qs}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ─── 2. Time-series Grid: Batch Size ─────────────────────────────────

def plot_timeseries_batch_grid(df, output_dir, q_name):
    patterns = [p for p in PATTERN_ORDER if p in df['load_pattern'].unique()]
    fig, axes = plt.subplots(len(patterns), 1, figsize=(14, 3.5 * len(patterns)),
                             sharex=False)
    if len(patterns) == 1:
        axes = [axes]

    for ax, pattern in zip(axes, patterns):
        for mode in CONTROLLER_ORDER:
            actual = resolve_controller(mode, q_name)
            sub = df[(df['controller_mode'] == actual) & (df['load_pattern'] == pattern)]
            sub = sub.sort_values('elapsed_sec')
            if len(sub) == 0:
                continue
            ax.plot(sub['elapsed_sec'], sub['batch_size'],
                    color=CONTROLLER_COLORS[mode], alpha=0.85, lw=1.2,
                    label=CONTROLLER_LABELS[mode])

        ax.set_title(PATTERN_LABELS[pattern], fontweight='bold')
        ax.set_ylabel('Batch Size')
        ax.set_xlabel('Time (s)')

    _controller_legend(axes[0])
    qs = q_short(q_name)
    fig.suptitle(f'Batch Size Over Time by Load Pattern ({qs})', fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, f'timeseries_batch_grid_{qs}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ─── 2b–2f. Time-series Grid: Other State & Control Variables ─────────

TIMESERIES_VARS = [
    ('cpu_util',           'CPU Utilization (%)',    'timeseries_cpu_grid'),
    ('container_mem_pct',  'Container Mem (%)',       'timeseries_mem_grid'),
    ('io_write_ops',       'I/O Write Ops',          'timeseries_io_grid'),
    ('avg_latency_ms',    'Avg Latency (ms)',          'timeseries_latency_grid'),
    ('poll_interval_ms',   'Poll Interval (ms)',      'timeseries_poll_grid'),
]


def plot_timeseries_variable_grid(df, output_dir, q_name, var_col, ylabel, file_prefix):
    """Generic time-series grid plot for any variable column."""
    patterns = [p for p in PATTERN_ORDER if p in df['load_pattern'].unique()]
    fig, axes = plt.subplots(len(patterns), 1, figsize=(14, 3.5 * len(patterns)),
                             sharex=False)
    if len(patterns) == 1:
        axes = [axes]

    for ax, pattern in zip(axes, patterns):
        for mode in CONTROLLER_ORDER:
            actual = resolve_controller(mode, q_name)
            sub = df[(df['controller_mode'] == actual) & (df['load_pattern'] == pattern)]
            sub = sub.sort_values('elapsed_sec')
            if len(sub) == 0 or var_col not in sub.columns:
                continue
            ax.plot(sub['elapsed_sec'], sub[var_col],
                    color=CONTROLLER_COLORS[mode], alpha=0.85, lw=1.2,
                    label=CONTROLLER_LABELS[mode])

        ax.set_title(PATTERN_LABELS[pattern], fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Time (s)')

    _controller_legend(axes[0])
    qs = q_short(q_name)
    fig.suptitle(f'{ylabel} Over Time by Load Pattern ({qs})', fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, f'{file_prefix}_{qs}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ─── 3. Bar Chart: Backlog Mean ──────────────────────────────────────

def plot_bar_backlog_mean(df, output_dir, q_name):
    patterns = [p for p in PATTERN_ORDER if p in df['load_pattern'].unique()]
    fig, axes = plt.subplots(1, len(patterns), figsize=(4 * len(patterns), 5), sharey=False)
    if len(patterns) == 1:
        axes = [axes]

    for ax, pattern in zip(axes, patterns):
        means = []
        colors = []
        labels = []
        for mode in CONTROLLER_ORDER:
            actual = resolve_controller(mode, q_name)
            sub = df[(df['controller_mode'] == actual) & (df['load_pattern'] == pattern)]
            means.append(sub['queue_length'].mean() if len(sub) > 0 else 0)
            colors.append(CONTROLLER_COLORS[mode])
            labels.append(CONTROLLER_LABELS[mode])

        bars = ax.bar(labels, means, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_title(PATTERN_LABELS[pattern], fontweight='bold')
        ax.set_ylabel('Mean Queue Length')
        ax.tick_params(axis='x', rotation=45)

        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:,.0f}', ha='center', va='bottom', fontsize=8)

    qs = q_short(q_name)
    fig.suptitle(f'Mean Backlog by Controller and Load Pattern ({qs})',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, f'bar_backlog_mean_{qs}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ─── 4. Bar Chart: Backlog Max ───────────────────────────────────────

def plot_bar_backlog_max(df, output_dir, q_name):
    patterns = [p for p in PATTERN_ORDER if p in df['load_pattern'].unique()]
    fig, axes = plt.subplots(1, len(patterns), figsize=(4 * len(patterns), 5), sharey=False)
    if len(patterns) == 1:
        axes = [axes]

    for ax, pattern in zip(axes, patterns):
        maxs = []
        colors = []
        labels = []
        for mode in CONTROLLER_ORDER:
            actual = resolve_controller(mode, q_name)
            sub = df[(df['controller_mode'] == actual) & (df['load_pattern'] == pattern)]
            maxs.append(int(sub['queue_length'].max()) if len(sub) > 0 else 0)
            colors.append(CONTROLLER_COLORS[mode])
            labels.append(CONTROLLER_LABELS[mode])

        bars = ax.bar(labels, maxs, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_title(PATTERN_LABELS[pattern], fontweight='bold')
        ax.set_ylabel('Max Queue Length')
        ax.tick_params(axis='x', rotation=45)

        for bar, val in zip(bars, maxs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:,}', ha='center', va='bottom', fontsize=8)

    qs = q_short(q_name)
    fig.suptitle(f'Peak Backlog by Controller and Load Pattern ({qs})',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, f'bar_backlog_max_{qs}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ─── 5. Bar Chart: Throughput ─────────────────────────────────────────

def plot_bar_throughput(df, output_dir, q_name):
    patterns = [p for p in PATTERN_ORDER if p in df['load_pattern'].unique()]
    fig, axes = plt.subplots(1, len(patterns), figsize=(4 * len(patterns), 5), sharey=False)
    if len(patterns) == 1:
        axes = [axes]

    for ax, pattern in zip(axes, patterns):
        throughputs = []
        colors = []
        labels = []
        for mode in CONTROLLER_ORDER:
            actual = resolve_controller(mode, q_name)
            sub = df[(df['controller_mode'] == actual) & (df['load_pattern'] == pattern)]
            sub = sub.sort_values('elapsed_sec')
            if len(sub) > 1:
                duration = sub['elapsed_sec'].max() - sub['elapsed_sec'].min()
                tp = sub['messages_indexed'].sum() / duration if duration > 0 else 0
            else:
                tp = 0
            throughputs.append(tp)
            colors.append(CONTROLLER_COLORS[mode])
            labels.append(CONTROLLER_LABELS[mode])

        bars = ax.bar(labels, throughputs, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_title(PATTERN_LABELS[pattern], fontweight='bold')
        ax.set_ylabel('Throughput (msg/s)')
        ax.tick_params(axis='x', rotation=45)

        for bar, val in zip(bars, throughputs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8)

    qs = q_short(q_name)
    fig.suptitle(f'Throughput by Controller and Load Pattern ({qs})',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, f'bar_throughput_{qs}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ─── 6. Bar Chart: Efficiency (CPU + Memory) ─────────────────────────

def plot_bar_efficiency(df, output_dir, q_name):
    patterns = [p for p in PATTERN_ORDER if p in df['load_pattern'].unique()]
    fig, axes = plt.subplots(2, len(patterns), figsize=(4 * len(patterns), 8), sharey='row')
    if len(patterns) == 1:
        axes = axes.reshape(2, 1)

    for col, pattern in enumerate(patterns):
        for idx, mode in enumerate(CONTROLLER_ORDER):
            actual = resolve_controller(mode, q_name)
            sub = df[(df['controller_mode'] == actual) & (df['load_pattern'] == pattern)]
            cpu_mean = sub['cpu_util'].mean() if len(sub) > 0 else 0
            mem_mean = sub['container_mem_pct'].mean() if (len(sub) > 0 and 'container_mem_pct' in sub.columns) else 0

            axes[0, col].bar(idx, cpu_mean, color=CONTROLLER_COLORS[mode],
                             edgecolor='white', linewidth=0.5)
            axes[1, col].bar(idx, mem_mean, color=CONTROLLER_COLORS[mode],
                             edgecolor='white', linewidth=0.5)

        axes[0, col].set_title(PATTERN_LABELS[pattern], fontweight='bold')
        axes[0, col].set_xticks(range(len(CONTROLLER_ORDER)))
        axes[0, col].set_xticklabels([CONTROLLER_LABELS[m] for m in CONTROLLER_ORDER],
                                      rotation=45, ha='right')
        axes[1, col].set_xticks(range(len(CONTROLLER_ORDER)))
        axes[1, col].set_xticklabels([CONTROLLER_LABELS[m] for m in CONTROLLER_ORDER],
                                      rotation=45, ha='right')

    axes[0, 0].set_ylabel('CPU Utilization (%)')
    axes[1, 0].set_ylabel('Memory Utilization (%)')

    qs = q_short(q_name)
    fig.suptitle(f'Resource Efficiency by Controller and Load Pattern ({qs})',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, f'bar_efficiency_{qs}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ─── 7. Heatmap Summary ──────────────────────────────────────────────

def plot_heatmap_summary(df, output_dir, q_name):
    metrics = {
        'Backlog Mean': lambda sub: sub['queue_length'].mean(),
        'Backlog Max': lambda sub: sub['queue_length'].max(),
        'Throughput (msg/s)': lambda sub: (
            sub['messages_indexed'].sum() /
            max(1, sub['elapsed_sec'].max() - sub['elapsed_sec'].min())
        ),
        'CPU Mean (%)': lambda sub: sub['cpu_util'].mean(),
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))

    for ax, (metric_name, metric_fn) in zip(axes, metrics.items()):
        data = np.zeros((len(CONTROLLER_ORDER), len(PATTERN_ORDER)))
        for i, mode in enumerate(CONTROLLER_ORDER):
            actual = resolve_controller(mode, q_name)
            for j, pattern in enumerate(PATTERN_ORDER):
                sub = df[(df['controller_mode'] == actual) & (df['load_pattern'] == pattern)]
                data[i, j] = metric_fn(sub) if len(sub) > 0 else np.nan

        sns.heatmap(data, ax=ax, annot=True, fmt='.0f', cmap='YlOrRd',
                    xticklabels=[PATTERN_LABELS[p].split(' ')[0] for p in PATTERN_ORDER],
                    yticklabels=[CONTROLLER_LABELS[m] for m in CONTROLLER_ORDER])
        ax.set_title(metric_name, fontweight='bold')

    qs = q_short(q_name)
    fig.suptitle(f'Performance Heatmap: Controller vs Load Pattern ({qs})',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, f'heatmap_summary_{qs}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ─── 8. Injection Rate vs Queue Length ────────────────────────────────

def plot_injection_vs_queue(df, output_dir, q_name):
    patterns = [p for p in PATTERN_ORDER if p in df['load_pattern'].unique()]
    fig, axes = plt.subplots(len(patterns), 1, figsize=(14, 4 * len(patterns)),
                             sharex=False)
    if len(patterns) == 1:
        axes = [axes]

    for ax, pattern in zip(axes, patterns):
        ax2 = ax.twinx()

        # Plot injection rate (use first controller's data — rate is same for all)
        first_actual = resolve_controller(CONTROLLER_ORDER[0], q_name)
        inj_sub = df[(df['controller_mode'] == first_actual) & (df['load_pattern'] == pattern)]
        inj_sub = inj_sub.sort_values('elapsed_sec')
        if len(inj_sub) > 0:
            ax2.fill_between(inj_sub['elapsed_sec'], inj_sub['injection_rate'],
                             alpha=0.15, color='gray', label='Injection Rate')
            ax2.set_ylabel('Injection Rate (rows/s)', color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')

        # Plot queue length for each controller
        for mode in CONTROLLER_ORDER:
            actual = resolve_controller(mode, q_name)
            sub = df[(df['controller_mode'] == actual) & (df['load_pattern'] == pattern)]
            sub = sub.sort_values('elapsed_sec')
            if len(sub) == 0:
                continue
            ax.plot(sub['elapsed_sec'], sub['queue_length'],
                    color=CONTROLLER_COLORS[mode], alpha=0.85, lw=1.2,
                    label=CONTROLLER_LABELS[mode])

        ax.set_title(PATTERN_LABELS[pattern], fontweight='bold')
        ax.set_ylabel('Queue Length')
        ax.set_xlabel('Time (s)')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    _controller_legend(axes[0])
    qs = q_short(q_name)
    fig.suptitle(f'Queue Length vs Load Injection Rate ({qs})',
                 fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, f'injection_vs_queue_{qs}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ─── 9. Boxplot: Queue Length Distribution ────────────────────────────

def plot_boxplot_queue(df, output_dir, q_name):
    patterns = [p for p in PATTERN_ORDER if p in df['load_pattern'].unique()]
    fig, axes = plt.subplots(1, len(patterns), figsize=(4.5 * len(patterns), 5), sharey=False)
    if len(patterns) == 1:
        axes = [axes]

    for ax, pattern in zip(axes, patterns):
        plot_data = []
        plot_labels = []
        plot_colors = []
        for mode in CONTROLLER_ORDER:
            actual = resolve_controller(mode, q_name)
            sub = df[(df['controller_mode'] == actual) & (df['load_pattern'] == pattern)]
            if len(sub) > 0:
                plot_data.append(sub['queue_length'].values)
                plot_labels.append(CONTROLLER_LABELS[mode])
                plot_colors.append(CONTROLLER_COLORS[mode])

        bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True,
                        showfliers=True, flierprops={'markersize': 2, 'alpha': 0.5})
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(PATTERN_LABELS[pattern], fontweight='bold')
        ax.set_ylabel('Queue Length')
        ax.tick_params(axis='x', rotation=45)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    qs = q_short(q_name)
    fig.suptitle(f'Queue Length Distribution by Controller and Load Pattern ({qs})',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, f'boxplot_queue_{qs}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ─── 10. Radar/Spider Chart ──────────────────────────────────────────

def plot_radar_comparison(df, output_dir, q_name):
    """Radar chart comparing controllers on normalized metrics per pattern."""
    metric_fns = {
        'Low Backlog':    lambda s: 1.0 / max(1, s['queue_length'].mean()),
        'Throughput':     lambda s: s['messages_indexed'].sum() / max(1, s['elapsed_sec'].max() - s['elapsed_sec'].min()),
        'Low CPU':        lambda s: 1.0 / max(0.1, s['cpu_util'].mean()),
        'Low Memory':     lambda s: 1.0 / max(0.1, s['container_mem_pct'].mean() if 'container_mem_pct' in s.columns else s['mem_util'].mean()),
        'Batch Adapt.':   lambda s: s['batch_size'].std() if len(s) > 1 else 0,
    }
    metric_names = list(metric_fns.keys())
    N = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    patterns = [p for p in PATTERN_ORDER if p in df['load_pattern'].unique()]
    fig, axes = plt.subplots(1, len(patterns), figsize=(5 * len(patterns), 5),
                             subplot_kw=dict(polar=True))
    if len(patterns) == 1:
        axes = [axes]

    for ax, pattern in zip(axes, patterns):
        # Compute raw values
        raw = {}
        for mode in CONTROLLER_ORDER:
            actual = resolve_controller(mode, q_name)
            sub = df[(df['controller_mode'] == actual) & (df['load_pattern'] == pattern)]
            if len(sub) == 0:
                continue
            raw[mode] = [fn(sub) for fn in metric_fns.values()]

        if not raw:
            continue

        # Normalize to [0, 1] per metric
        all_vals = np.array(list(raw.values()))
        mins = all_vals.min(axis=0)
        maxs = all_vals.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1

        for mode, vals in raw.items():
            normed = ((np.array(vals) - mins) / ranges).tolist()
            normed += normed[:1]
            ax.plot(angles, normed, color=CONTROLLER_COLORS[mode], lw=1.5,
                    label=CONTROLLER_LABELS[mode])
            ax.fill(angles, normed, color=CONTROLLER_COLORS[mode], alpha=0.08)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=8)
        ax.set_title(PATTERN_LABELS[pattern], fontweight='bold', pad=20)
        ax.set_ylim(0, 1.1)

    axes[-1].legend(loc='lower right', bbox_to_anchor=(1.3, -0.1), fontsize=8)
    qs = q_short(q_name)
    fig.suptitle(f'Normalized Controller Comparison — Radar ({qs})',
                 fontsize=15, fontweight='bold', y=1.05)
    fig.tight_layout()
    path = os.path.join(output_dir, f'radar_comparison_{qs}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ─── 11. Bar Chart: Cost J per Q Config ───────────────────────────────

Q_LABELS = {
    'Q1_backlog':  'Q1 (Backlog)',
    'Q2_resource': 'Q2 (Resource)',
    'Q4_balanced': 'Q4 (Balanced)',
}
Q_ORDER = ['Q1_backlog', 'Q2_resource', 'Q4_balanced']


def plot_bar_cost_J(analysis, output_dir):
    """Grouped bar chart: cost J per controller (5 matched), one subplot per Q config."""
    sensitivity = analysis.get('sensitivity', {})
    cost_table = sensitivity.get('cost_table', {})
    if not cost_table:
        print("  Skipped bar_cost_J.png (no cost data)")
        return

    q_names = [q for q in Q_ORDER if q in cost_table]
    n_ctrl = len(CONTROLLER_ORDER)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    axes = axes.flatten()

    for idx, q_name in enumerate(q_names):
        ax = axes[idx]
        q_costs = cost_table[q_name]
        vals = []
        colors = []
        labels = []
        for mode in CONTROLLER_ORDER:
            vals.append(q_costs.get(mode, 0))
            colors.append(CONTROLLER_COLORS[mode])
            labels.append(CONTROLLER_LABELS[mode])

        x = np.arange(n_ctrl)
        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.5, width=0.6)
        ax.set_title(Q_LABELS.get(q_name, q_name), fontweight='bold')
        ax.set_ylabel('Cost J (mean across patterns)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:,.0f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle('Cost Function J by Controller and Q Configuration (Thesis §III.5.4)',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, 'bar_cost_J.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ─── 12. Heatmap: Cost J (Controller × Q Config) ─────────────────────

def plot_heatmap_cost_J(analysis, output_dir):
    """Heatmap of mean cost J: rows=5 controllers, cols=Q configs."""
    sensitivity = analysis.get('sensitivity', {})
    cost_table = sensitivity.get('cost_table', {})
    if not cost_table:
        print("  Skipped heatmap_cost_J.png (no cost data)")
        return

    q_names = [q for q in Q_ORDER if q in cost_table]
    data = np.zeros((len(CONTROLLER_ORDER), len(q_names)))

    for i, mode in enumerate(CONTROLLER_ORDER):
        for j, q_name in enumerate(q_names):
            data[i, j] = cost_table[q_name].get(mode, np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(data, ax=ax, annot=True, fmt='.0f', cmap='YlOrRd',
                xticklabels=[Q_LABELS.get(q, q) for q in q_names],
                yticklabels=[CONTROLLER_LABELS[m] for m in CONTROLLER_ORDER],
                annot_kws={'fontsize': 10})
    ax.set_title('Cost Function J: Controller vs Q Configuration',
                 fontweight='bold', fontsize=13)
    ax.set_xlabel('Q Configuration')
    ax.set_ylabel('Controller')

    fig.tight_layout()
    path = os.path.join(output_dir, 'heatmap_cost_J.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ─── 13. Bar Chart: Normalized Regret (Thesis §III.5.5) ──────────────

def plot_bar_regret(analysis, output_dir):
    """Normalized regret bar chart: 5 controllers, grouped bars per Q config."""
    sensitivity = analysis.get('sensitivity', {})
    regret = sensitivity.get('regret', {})
    if not regret:
        print("  Skipped bar_regret.png (no regret data)")
        return

    q_names = [q for q in Q_ORDER if q in sensitivity.get('cost_table', {})]
    n_q = len(q_names)
    n_ctrl = len(CONTROLLER_ORDER)
    bar_width = 0.18
    x = np.arange(n_ctrl)

    fig, ax = plt.subplots(figsize=(12, 6))

    q_colors = ['#2196F3', '#FF9800', '#9C27B0', '#4CAF50']

    for j, q_name in enumerate(q_names):
        vals = [regret.get(mode, {}).get(q_name, 0) for mode in CONTROLLER_ORDER]
        offset = (j - n_q / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width, label=Q_LABELS.get(q_name, q_name),
               color=q_colors[j % len(q_colors)], alpha=0.8, edgecolor='white')

    # Add mean/max regret markers
    mean_regrets = [regret.get(mode, {}).get('mean_regret', 0) for mode in CONTROLLER_ORDER]
    max_regrets = [regret.get(mode, {}).get('max_regret', 0) for mode in CONTROLLER_ORDER]

    ax.plot(x, mean_regrets, 'ko-', markersize=8, linewidth=1.5, label='Mean Regret', zorder=5)
    ax.plot(x, max_regrets, 'r^--', markersize=8, linewidth=1.5, label='Max Regret', zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([CONTROLLER_LABELS[m] for m in CONTROLLER_ORDER], fontsize=10)
    ax.set_ylabel('Normalized Regret')
    ax.set_title('Normalized Regret by Controller and Q Configuration (Thesis Eq. 3.13-3.15)',
                 fontweight='bold', fontsize=13)
    ax.legend(loc='upper right', fontsize=9)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    # Annotate mean regret values
    for i, mr in enumerate(mean_regrets):
        ax.annotate(f'{mr:.4f}', (x[i], mr), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=8, color='black')

    fig.tight_layout()
    path = os.path.join(output_dir, 'bar_regret.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ─── 14. Cost J Breakdown per Load Pattern ────────────────────────────

def plot_bar_cost_by_pattern(analysis, output_dir):
    """Cost J per pattern: one figure per Q config, 5 matched controllers each."""
    per_test = analysis.get('per_test', {})
    if not per_test:
        print("  Skipped bar_cost_by_pattern (no per-test data)")
        return

    patterns = [p for p in PATTERN_ORDER if p in per_test]
    n_ctrl = len(CONTROLLER_ORDER)

    for q_name in Q_ORDER:
        matched = Q_MATCHED_CONTROLLERS[q_name]
        controller_map = {
            **{c: c for c in BASELINE_CONTROLLERS},
            'state_fb': 'state_fb',
            'lqr': matched['lqr'],
            'ann': matched['ann'],
        }

        n_pat = len(patterns)
        n_cols = min(n_pat, 3)
        n_rows = (n_pat + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharey=False)
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for idx, pattern in enumerate(patterns):
            ax = axes_flat[idx]
            vals = []
            colors = []
            labels = []
            for mode in CONTROLLER_ORDER:
                actual_mode = controller_map[mode]
                mode_data = per_test[pattern].get(actual_mode)
                if mode_data and 'costs' in mode_data:
                    vals.append(mode_data['costs'].get(q_name, 0))
                else:
                    vals.append(0)
                colors.append(CONTROLLER_COLORS[mode])
                labels.append(CONTROLLER_LABELS[mode])

            x = np.arange(n_ctrl)
            bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.5, width=0.6)
            ax.set_title(PATTERN_LABELS.get(pattern, pattern), fontweight='bold')
            ax.set_ylabel(f'Cost J ({Q_LABELS.get(q_name, q_name)})')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)

            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:,.0f}', ha='center', va='bottom', fontsize=8)

        qs = q_short(q_name)
        fig.suptitle(f'Cost J ({Q_LABELS.get(q_name, q_name)}) by Controller and Load Pattern',
                     fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()
        path = os.path.join(output_dir, f'bar_cost_by_pattern_{qs}.png')
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")


# ─── Cost Decomposition per Variable ──────────────────────────────────

def plot_cost_decomposition(df, output_dir, normalization=None):
    """Bar chart: cost contribution per variable for each controller, per Q config."""
    ONE_SIDED = [i for i, v in enumerate(STATE_VARS) if v in ('cpu_util', 'container_mem_pct', 'io_write_ops')]
    ctrl_norm_keys = ['batch_size', 'inv_poll_interval']
    labels = [v.replace('_', '\n') for v in STATE_VARS] + ['u_batch', 'u_poll']
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336', '#607D8B', '#795548']

    norm = normalization or {}

    def compute_costs(sub, Q):
        costs = []
        for i, var in enumerate(STATE_VARS):
            x_raw = sub[var].values.astype(float)
            if var in norm:
                x_n = (x_raw - norm[var]['mean']) / norm[var]['std']
                t_n = (X_TARGET[i] - norm[var]['mean']) / norm[var]['std']
            else:
                x_n = x_raw; t_n = X_TARGET[i]
            err = x_n - t_n
            if i in ONE_SIDED:
                err = np.maximum(0, err)
            costs.append(Q[i, i] * np.sum(err ** 2))
        for i, var in enumerate(CONTROL_VARS):
            key = ctrl_norm_keys[i]
            col = var if var in sub.columns else 'poll_interval_ms'
            u = sub[col].values.astype(float)
            if key in norm:
                u = (u - norm[key]['mean']) / norm[key]['std']
            costs.append(R_MATRIX[i, i] * np.sum(u ** 2))
        return costs

    for q_name, Q in Q_CONFIGS.items():
        qs = q_short(q_name)
        n_ctrl = len(CONTROLLER_ORDER)
        fig, axes = plt.subplots(1, n_ctrl, figsize=(4 * n_ctrl, 5), sharey=True)
        if n_ctrl == 1:
            axes = [axes]
        fig.suptitle(f'Cost Decomposition - {Q_LABELS.get(q_name, q_name)} (Q={list(np.diag(Q).astype(int))})',
                     fontsize=14, fontweight='bold')

        for idx, ctrl in enumerate(CONTROLLER_ORDER):
            ax = axes[idx]
            actual_mode = resolve_controller(ctrl, q_name)
            sub = df[df['controller_mode'] == actual_mode]
            if len(sub) == 0:
                ax.set_title(ctrl)
                continue

            w = compute_costs(sub, Q)
            total = sum(w)
            pcts = [v / total * 100 if total > 0 else 0 for v in w]

            bars = ax.bar(range(len(pcts)), pcts, color=colors[:len(pcts)])
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.set_title(f'{CONTROLLER_LABELS[ctrl]} (J={total:.0f})', fontweight='bold')
            if idx == 0:
                ax.set_ylabel('% of Total J')

            for bar, val in zip(bars, w):
                if val > 0.005 * total:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f'{val:.0f}', ha='center', va='bottom', fontsize=7)

        fig.tight_layout()
        path = os.path.join(output_dir, f'cost_decomposition_{qs}.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Closed-loop experiment visualization')
    parser.add_argument('--csv', required=True, help='Path to closed_loop_data.csv')
    parser.add_argument('--sysid-json', default=None,
                        help='Path to sysid JSON for cost normalization')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for plots (default: same as CSV)')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} samples from {args.csv}")
    print(f"  Controllers: {df['controller_mode'].unique().tolist()}")
    print(f"  Patterns: {df['load_pattern'].unique().tolist()}")

    output_dir = args.output_dir or os.path.dirname(args.csv)
    os.makedirs(output_dir, exist_ok=True)

    # Load normalization for cost computation
    normalization = None
    if args.sysid_json:
        with open(args.sysid_json) as f:
            sysid = json.load(f)
        normalization = sysid.get('normalization', None)
        print(f"  Loaded normalization from {args.sysid_json}")

    # Generate performance plots per Q config (×4)
    for q_name in Q_ORDER:
        qs = q_short(q_name)
        print(f"\nGenerating performance plots for {qs}...")
        plot_timeseries_queue_grid(df, output_dir, q_name)
        plot_timeseries_batch_grid(df, output_dir, q_name)
        for var_col, ylabel, file_prefix in TIMESERIES_VARS:
            plot_timeseries_variable_grid(df, output_dir, q_name, var_col, ylabel, file_prefix)
        plot_bar_backlog_mean(df, output_dir, q_name)
        plot_bar_backlog_max(df, output_dir, q_name)
        plot_bar_throughput(df, output_dir, q_name)
        plot_bar_efficiency(df, output_dir, q_name)
        plot_heatmap_summary(df, output_dir, q_name)
        plot_injection_vs_queue(df, output_dir, q_name)
        plot_boxplot_queue(df, output_dir, q_name)
        plot_radar_comparison(df, output_dir, q_name)

    # Cost & regret plots (thesis §III.5.4–§III.5.5) — aggregate
    print("\nComputing cost function J and sensitivity analysis...")
    analysis = analyze_experiment(df, normalization=normalization)

    print("\nGenerating cost & regret plots...")
    plot_bar_cost_J(analysis, output_dir)
    plot_heatmap_cost_J(analysis, output_dir)
    plot_bar_regret(analysis, output_dir)
    plot_bar_cost_by_pattern(analysis, output_dir)

    print("\nGenerating cost decomposition plots...")
    plot_cost_decomposition(df, output_dir, normalization=normalization)

    n_perf = (10 + len(TIMESERIES_VARS)) * len(Q_ORDER)
    n_cost = 4 + len(Q_ORDER) * 2  # cost_J, heatmap, regret + cost_by_pattern + decomposition
    print(f"\nAll {n_perf + n_cost} plots saved to {output_dir}")


if __name__ == '__main__':
    main()
