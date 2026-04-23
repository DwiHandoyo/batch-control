"""
Metrics Analysis — Compute thesis-aligned metrics from closed-loop experiment data.

Metrics (thesis chapter-3 §III.5):
  A. Core Sync: backlog mean/max/recovery_time
  B. Control Dynamics: rise_time, overshoot, settling_time
  C. Processing Efficiency: throughput, cpu_efficiency, mem_efficiency
  D. Cost Function: J = Σ(x'Qx + u'Ru)
  E. Sensitivity: normalized regret across Q configurations

Usage:
    python metrics_analysis.py --csv <closed_loop_data.csv> [--sysid-json <path>]
    python metrics_analysis.py --csv <path> --output-dir <dir>
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('metrics-analysis')

STATE_VARS = ['queue_length', 'cpu_util', 'container_mem_pct', 'io_write_ops']
CONTROL_VARS = ['batch_size', 'inv_poll_interval']

# Target state (desired equilibrium) — consistent with controllers.py x_target
X_TARGET = np.array([0.0, 5.0, 70.0, 30.0])  # container_mem_pct target

# Q matrix configurations (4-dim state: [queue, cpu, mem, io])
# MUST match controllers.py Q_PRESETS exactly
Q_CONFIGS = {
    'Q1_backlog':  np.diag([20.0,  5.0, 0.05,  1.0]),  # backlog priority
    'Q2_resource': np.diag([10.0, 10.0, 0.10,  2.0]),  # resource priority
    'Q4_balanced': np.diag([20.0, 10.0, 0.10,  2.0]),  # balanced aggressive
}

R_MATRIX = np.diag([1e-6, 1e-6])  # near-zero: no control effort penalty

# Matched controller mapping: for each Q config, which LQR/ANN variant to use.
# ANN uses universal MLE model (ann_cw_*); falls back to specialized ann_* if present.
Q_MATCHED_CONTROLLERS = {
    'Q1_backlog':  {'lqr': 'lqr_q1', 'ann': 'ann_cw_q1'},
    'Q2_resource': {'lqr': 'lqr_q2', 'ann': 'ann_cw_q2'},
    'Q4_balanced': {'lqr': 'lqr_q4', 'ann': 'ann_cw_q4'},
}
# 'pid' slot is filled by state_fb (pole-placement state feedback)
BASELINE_CONTROLLERS = ['static', 'rule_based', 'pid']
GENERIC_CONTROLLER_ORDER = ['static', 'rule_based', 'pid', 'lqr', 'ann']


# ─── Control Dynamics ─────────────────────────────────────────────────

def compute_control_dynamics(
    df: pd.DataFrame,
    reference: float = 0.0,
    tolerance: float = 500.0,
) -> Dict:
    """
    Compute rise time, overshoot, and settling time from a single test run.

    Args:
        df: DataFrame with 'elapsed_sec' and 'queue_length' columns, sorted by elapsed_sec
        reference: target queue_length (usually 0)
        tolerance: tolerance band for settling/rise time

    Returns:
        dict with rise_time, overshoot, settling_time (None if not reached)
    """
    queue = df['queue_length'].values.astype(float)
    elapsed = df['elapsed_sec'].values.astype(float)

    if len(queue) == 0:
        return {'rise_time': None, 'overshoot': None, 'settling_time': None}

    peak_idx = np.argmax(queue)
    peak_val = float(queue[peak_idx])

    # Overshoot: max backlog above reference (absolute and percentage)
    overshoot_abs = peak_val - reference
    if reference > 0:
        overshoot_pct = (peak_val - reference) / reference * 100
    else:
        overshoot_pct = peak_val  # absolute value when reference is 0

    # Rise time: time from peak until queue first drops below reference + tolerance
    rise_time = None
    for i in range(peak_idx, len(queue)):
        if queue[i] <= reference + tolerance:
            rise_time = float(elapsed[i] - elapsed[peak_idx])
            break

    # Settling time: last time queue goes outside tolerance band
    # Scan from end backward to find last excursion
    settling_time = None
    for i in range(len(queue) - 1, -1, -1):
        if abs(queue[i] - reference) > tolerance:
            settling_time = float(elapsed[i] - elapsed[0])
            break
    # If never outside tolerance, settling_time = 0
    if settling_time is None:
        settling_time = 0.0

    return {
        'rise_time': rise_time,
        'overshoot_abs': overshoot_abs,
        'overshoot_pct': overshoot_pct,
        'settling_time': settling_time,
        'peak_backlog': peak_val,
        'peak_time': float(elapsed[peak_idx]) if peak_idx < len(elapsed) else None,
    }


# ─── Core Sync Metrics ───────────────────────────────────────────────

def compute_sync_metrics(df: pd.DataFrame, reference: float = 0.0,
                         tolerance: float = 500.0) -> Dict:
    """Compute backlog mean, max, and recovery time."""
    queue = df['queue_length'].values.astype(float)
    elapsed = df['elapsed_sec'].values.astype(float)

    backlog_mean = float(np.mean(queue))
    backlog_max = float(np.max(queue))

    # Recovery time: time from max backlog to first return below tolerance
    peak_idx = np.argmax(queue)
    recovery_time = None
    for i in range(peak_idx, len(queue)):
        if queue[i] <= reference + tolerance:
            recovery_time = float(elapsed[i] - elapsed[peak_idx])
            break

    return {
        'backlog_mean': backlog_mean,
        'backlog_max': backlog_max,
        'backlog_recovery_time': recovery_time,
    }


# ─── Processing Efficiency ───────────────────────────────────────────

def compute_efficiency(df: pd.DataFrame) -> Dict:
    """Compute throughput and efficiency metrics."""
    duration = df['elapsed_sec'].max() - df['elapsed_sec'].min()
    total_indexed = df['messages_indexed'].sum()

    throughput = float(total_indexed / duration) if duration > 0 else 0.0

    cpu_mean = df['cpu_util'].mean()
    mem_mean = df['container_mem_pct'].mean() if 'container_mem_pct' in df.columns else df.get('mem_util', df['queue_length'] * 0).mean()

    cpu_efficiency = float(throughput / cpu_mean) if cpu_mean > 0 else 0.0
    mem_efficiency = float(throughput / mem_mean) if mem_mean > 0 else 0.0

    return {
        'throughput': throughput,
        'total_indexed': int(total_indexed),
        'cpu_mean': float(cpu_mean),
        'mem_mean': float(mem_mean),
        'io_mean': float(df['io_write_ops'].mean()),
        'cpu_efficiency': cpu_efficiency,
        'mem_efficiency': mem_efficiency,
    }


# ─── Cost Function ───────────────────────────────────────────────────

def compute_cost_J(
    df: pd.DataFrame,
    Q: np.ndarray,
    R: np.ndarray,
    normalization: Optional[Dict] = None,
    reference: Optional[np.ndarray] = None,
) -> float:
    """
    Compute quadratic cost J = Σ(x'Qx + u'Ru) over the trajectory.

    Args:
        df: DataFrame with STATE_VARS and CONTROL_VARS columns
        Q: state cost matrix (4×4)
        R: control cost matrix (2×2)
        normalization: optional dict with per-variable {mean, std} for z-score
        reference: state reference vector (default: zeros)
    """
    states = df[STATE_VARS].values.astype(float)
    controls = df[CONTROL_VARS].values.astype(float)

    if reference is None:
        ref = np.zeros(len(STATE_VARS))
    else:
        ref = np.array(reference, dtype=float)  # copy — do NOT mutate caller's array

    # Normalize if scales provided
    if normalization:
        for i, key in enumerate(STATE_VARS):
            if key in normalization:
                mean = normalization[key].get('mean', 0)
                std = normalization[key].get('std', 1)
                if std > 0:
                    states[:, i] = (states[:, i] - mean) / std
                    ref[i] = (ref[i] - mean) / std

        for i, key in enumerate(CONTROL_VARS):
            if key in normalization:
                mean = normalization[key].get('mean', 0)
                std = normalization[key].get('std', 1)
                if std > 0:
                    controls[:, i] = (controls[:, i] - mean) / std

    # One-sided penalty indices: cpu(1), mem(2), io(3) — only penalize overshoot
    ONE_SIDED_INDICES = [i for i, v in enumerate(STATE_VARS) if v in ('cpu_util', 'container_mem_pct', 'io_write_ops')]

    # Compute cost
    J = 0.0
    for t in range(len(states)):
        x = states[t] - ref
        # One-sided: zero out error for resource vars below target
        for i in ONE_SIDED_INDICES:
            x[i] = max(0.0, x[i])
        u = controls[t]
        J += float(x @ Q @ x + u @ R @ u)

    return J


# ─── Sensitivity Analysis ────────────────────────────────────────────

def compute_regret_table(
    cost_table: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute normalized regret from a cost table.

    Args:
        cost_table: {q_config: {controller_mode: J_value}}

    Returns:
        {controller_mode: {q_config: regret, 'mean_regret': ..., 'max_regret': ...}}
    """
    controllers = set()
    for q_costs in cost_table.values():
        controllers.update(q_costs.keys())
    controllers = sorted(controllers)

    regret = {c: {} for c in controllers}

    for q_name, q_costs in cost_table.items():
        j_min = min(q_costs.values()) if q_costs else 1.0
        j_min = max(j_min, 1e-10)  # avoid division by zero

        for controller in controllers:
            j_val = q_costs.get(controller, float('inf'))
            regret[controller][q_name] = (j_val - j_min) / j_min

    # Aggregate
    for controller in controllers:
        regret_vals = [v for k, v in regret[controller].items()
                       if k in cost_table]
        regret[controller]['mean_regret'] = float(np.mean(regret_vals)) if regret_vals else 0.0
        regret[controller]['max_regret'] = float(np.max(regret_vals)) if regret_vals else 0.0

    return regret


# ─── Full Analysis Pipeline ──────────────────────────────────────────

def analyze_experiment(
    df: pd.DataFrame,
    normalization: Optional[Dict] = None,
    reference: float = 0.0,
    tolerance: float = 500.0,
) -> Dict:
    """
    Run full analysis on closed-loop experiment data.

    Returns nested dict: {load_pattern: {controller_mode: {metric: value}}}
    Plus a top-level 'sensitivity' key with regret analysis.
    """
    results = {}

    patterns = df['load_pattern'].unique()
    modes = df['controller_mode'].unique()

    # Per-pattern, per-controller metrics
    for pattern in patterns:
        results[pattern] = {}
        for mode in modes:
            mdf = df[(df['load_pattern'] == pattern) &
                     (df['controller_mode'] == mode)].copy()
            mdf = mdf.sort_values('elapsed_sec')

            if len(mdf) == 0:
                results[pattern][mode] = None
                continue

            sync = compute_sync_metrics(mdf, reference, tolerance)
            dynamics = compute_control_dynamics(mdf, reference, tolerance)
            efficiency = compute_efficiency(mdf)

            # Cost for each Q config
            costs = {}
            for q_name, Q in Q_CONFIGS.items():
                costs[q_name] = compute_cost_J(
                    mdf, Q, R_MATRIX, normalization, reference=X_TARGET
                )

            results[pattern][mode] = {
                **sync,
                **dynamics,
                **efficiency,
                'costs': costs,
                'samples': len(mdf),
            }

    # Sensitivity analysis: aggregate costs across all patterns
    # For each Q config, only include matched controllers (5 per Q):
    #   static, rule_based, pid, lqr(K_Q), ann(model_Q)
    cost_table = {q_name: {} for q_name in Q_CONFIGS}
    for q_name in Q_CONFIGS:
        matched = Q_MATCHED_CONTROLLERS[q_name]
        # Map generic name → actual mode in experiment data
        controller_map = {
            **{c: c for c in BASELINE_CONTROLLERS},
            'lqr': matched['lqr'],
            'ann': matched['ann'],
        }
        for generic_name, actual_mode in controller_map.items():
            total_j = 0.0
            count = 0
            for pattern in patterns:
                if results[pattern].get(actual_mode) and results[pattern][actual_mode].get('costs'):
                    total_j += results[pattern][actual_mode]['costs'][q_name]
                    count += 1
            if count > 0:
                cost_table[q_name][generic_name] = total_j / count  # mean J across patterns

    regret = compute_regret_table(cost_table)

    return {
        'per_test': results,
        'sensitivity': {
            'cost_table': cost_table,
            'regret': regret,
            'q_configs': {k: v.tolist() for k, v in Q_CONFIGS.items()},
            'r_matrix': R_MATRIX.tolist(),
        },
    }


def format_report(analysis: Dict, test_duration: int = 300,
                  base_rate: int = 300) -> str:
    """Format analysis results as a human-readable report."""
    lines = []
    lines.append("=" * 100)
    lines.append("  CLOSED-LOOP EXPERIMENT — FULL METRICS REPORT")
    lines.append(f"  Generated: {datetime.utcnow().isoformat()}")
    lines.append("=" * 100)

    per_test = analysis['per_test']

    for pattern, mode_results in per_test.items():
        lines.append(f"\n{'=' * 100}")
        lines.append(f"  LOAD PATTERN: {pattern.upper()}")
        lines.append(f"{'=' * 100}")

        modes = [m for m in mode_results if mode_results[m] is not None]
        if not modes:
            lines.append("  No data")
            continue

        # Header
        header = f"{'Metric':<30s}"
        for mode in modes:
            header += f" {mode:>14s}"
        lines.append(header)
        lines.append("-" * (30 + 15 * len(modes)))

        def row(name, key, fmt=".2f"):
            r = f"{name:<30s}"
            for mode in modes:
                val = mode_results[mode].get(key)
                if val is None:
                    r += f" {'N/A':>14s}"
                elif fmt == "d":
                    r += f" {int(val):>14d}"
                else:
                    r += f" {val:>14{fmt}}"
            lines.append(r)

        lines.append("  --- Synchronization ---")
        row("Backlog mean", 'backlog_mean')
        row("Backlog max", 'backlog_max', 'd')
        row("Recovery time (s)", 'backlog_recovery_time')

        lines.append("  --- Control Dynamics ---")
        row("Rise time (s)", 'rise_time')
        row("Overshoot (abs)", 'overshoot_abs')
        row("Overshoot (%)", 'overshoot_pct')
        row("Settling time (s)", 'settling_time')
        row("Peak backlog", 'peak_backlog', 'd')

        lines.append("  --- Efficiency ---")
        row("Throughput (msg/s)", 'throughput')
        row("CPU mean (%)", 'cpu_mean')
        row("Mem mean (%)", 'mem_mean')
        row("IO mean", 'io_mean')
        row("CPU efficiency", 'cpu_efficiency')
        row("Mem efficiency", 'mem_efficiency')

        lines.append("  --- Cost Function J ---")
        for q_name in Q_CONFIGS:
            r = f"  {q_name:<28s}"
            for mode in modes:
                costs = mode_results[mode].get('costs', {})
                val = costs.get(q_name)
                if val is None:
                    r += f" {'N/A':>14s}"
                else:
                    r += f" {val:>14.2f}"
            lines.append(r)

    # Sensitivity / Regret Analysis (5 matched controllers per Q)
    sensitivity = analysis.get('sensitivity', {})
    regret = sensitivity.get('regret', {})
    if regret:
        lines.append(f"\n{'=' * 100}")
        lines.append("  SENSITIVITY ANALYSIS — NORMALIZED REGRET (5 Matched Controllers per Q)")
        lines.append(f"{'=' * 100}")

        q_names = list(Q_CONFIGS.keys())
        controller_labels = {
            'static': 'Static', 'rule_based': 'Rule-Based', 'pid': 'PID',
            'lqr': 'LQR', 'ann': 'ANN',
        }

        header = f"{'Controller':<18s}"
        for q_name in q_names:
            header += f" {q_name:>14s}"
        header += f" {'Mean Regret':>14s} {'Max Regret':>14s}"
        lines.append(header)
        lines.append("-" * (18 + 15 * (len(q_names) + 2)))

        for controller in GENERIC_CONTROLLER_ORDER:
            if controller not in regret:
                continue
            label = controller_labels.get(controller, controller)
            r = f"{label:<18s}"
            for q_name in q_names:
                val = regret[controller].get(q_name, 0)
                r += f" {val:>14.4f}"
            r += f" {regret[controller].get('mean_regret', 0):>14.4f}"
            r += f" {regret[controller].get('max_regret', 0):>14.4f}"
            lines.append(r)

    return '\n'.join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compute thesis metrics from closed-loop experiment data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python metrics_analysis.py --csv runs/closed_loop_20260324/results/closed_loop_data.csv
  python metrics_analysis.py --csv <path> --sysid-json <path> --tolerance 300
        """
    )
    parser.add_argument('--csv', required=True,
                        help='Path to closed_loop_data.csv')
    parser.add_argument('--sysid-json', default=None,
                        help='Path to sysid JSON for normalization scales')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: same as CSV)')
    parser.add_argument('--reference', type=float, default=0.0,
                        help='Queue reference value (default: 0)')
    parser.add_argument('--tolerance', type=float, default=500.0,
                        help='Tolerance band for settling/rise time (default: 500)')

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from {args.csv}")
    df = pd.read_csv(args.csv)
    logger.info(f"Loaded {len(df)} samples, "
                f"patterns={df['load_pattern'].unique().tolist()}, "
                f"modes={df['controller_mode'].unique().tolist()}")

    # Load normalization
    normalization = None
    if args.sysid_json:
        with open(args.sysid_json) as f:
            sysid = json.load(f)
        normalization = sysid.get('normalization', None)
        logger.info("Loaded normalization scales from sysid JSON")

    # Run analysis
    analysis = analyze_experiment(
        df, normalization=normalization,
        reference=args.reference, tolerance=args.tolerance,
    )

    # Output
    output_dir = args.output_dir or os.path.dirname(args.csv)
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    json_path = os.path.join(output_dir, 'metrics_summary.json')

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=convert)
    logger.info(f"Metrics JSON saved to {json_path}")

    # Save report
    report = format_report(analysis)
    report_path = os.path.join(output_dir, 'metrics_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

    print(f"\n{report}")


if __name__ == '__main__':
    main()
