"""
Run Selector — CLI tool for selecting experiment datasets for LQR and ANN training.

Usage:
    python select_run.py list                              # List all runs with data
    python select_run.py inspect <run_id>                  # Show run details
    python select_run.py sysid <run_id> [--normalize]      # Run sysid on one run
    python select_run.py sysid --runs <id1> <id2> [opts]   # Merge runs then sysid
"""

import os
import sys
import json
import glob
import argparse
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('select-run')

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')


@dataclass
class RunInfo:
    """Summary of an experiment run."""
    run_id: str
    path: str
    grid_preset: str = ''
    grid_size: str = ''
    batch_range: str = ''
    poll_range: str = ''
    phases: List[str] = field(default_factory=list)
    total_steps: int = 0
    step_duration: int = 0
    csv_files: List[str] = field(default_factory=list)
    csv_total_size: int = 0
    sample_count: int = 0
    has_sysid: bool = False
    sysid_json: Optional[str] = None
    sysid_r_squared: Optional[dict] = None


def scan_run(run_dir: str) -> Optional[RunInfo]:
    """Scan a run directory and extract metadata + data info."""
    run_id = os.path.basename(run_dir)
    if not run_id.startswith('run_'):
        return None

    run_id = run_id[4:]  # strip 'run_' prefix
    info = RunInfo(run_id=run_id, path=run_dir)

    # Read metadata
    meta_path = os.path.join(run_dir, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        gc = meta.get('grid_config', {})
        info.grid_preset = gc.get('preset', '')
        batch_vals = gc.get('batch_size_values', [])
        poll_vals = gc.get('poll_interval_values', [])
        info.grid_size = f"{len(batch_vals)}x{len(poll_vals)}" if batch_vals and poll_vals else ''
        info.batch_range = f"{gc.get('batch_min', '?')}-{gc.get('batch_max', '?')}" if gc else ''
        info.poll_range = f"{gc.get('poll_min', '?')}-{gc.get('poll_max', '?')}" if gc else ''
        ep = meta.get('experiment_params', {})
        info.phases = ep.get('phases', [])
        info.total_steps = ep.get('total_steps', 0)
        info.step_duration = ep.get('step_duration', 0)

    # Find CSV files
    results_dir = os.path.join(run_dir, 'results')
    if os.path.isdir(results_dir):
        csvs = glob.glob(os.path.join(results_dir, '*.csv'))
        info.csv_files = csvs
        info.csv_total_size = sum(os.path.getsize(f) for f in csvs)
        # Count samples from first CSV
        if csvs:
            try:
                df = pd.read_csv(csvs[0], usecols=['phase'])
                info.sample_count = len(df)
            except Exception:
                pass

    # Check sysid output
    sysid_dir = os.path.join(results_dir, 'sysid_output') if os.path.isdir(results_dir) else ''
    if sysid_dir and os.path.isdir(sysid_dir):
        jsons = glob.glob(os.path.join(sysid_dir, 'sysid_matrices_*.json'))
        if jsons:
            info.has_sysid = True
            info.sysid_json = sorted(jsons)[-1]  # latest
            try:
                with open(info.sysid_json) as f:
                    sdata = json.load(f)
                r2 = sdata.get('fit_metrics', {}).get('r_squared', [])
                svars = sdata.get('state_vars', [])
                if r2 and svars:
                    info.sysid_r_squared = dict(zip(svars, r2))
            except Exception:
                pass

    return info


def list_runs(runs_dir: str = RUNS_DIR) -> List[RunInfo]:
    """Scan all runs and return info for those with CSV data."""
    runs = []
    if not os.path.isdir(runs_dir):
        return runs
    for entry in sorted(os.listdir(runs_dir)):
        run_path = os.path.join(runs_dir, entry)
        if not os.path.isdir(run_path) or not entry.startswith('run_'):
            continue
        info = scan_run(run_path)
        if info and info.csv_files:
            runs.append(info)
    return runs


def get_run_info(run_id: str, runs_dir: str = RUNS_DIR) -> RunInfo:
    """Get RunInfo for a specific run ID."""
    run_path = os.path.join(runs_dir, f'run_{run_id}')
    if not os.path.isdir(run_path):
        raise FileNotFoundError(f"Run not found: {run_path}")
    info = scan_run(run_path)
    if not info:
        raise ValueError(f"Could not parse run: {run_id}")
    return info


# Column renames for backward compatibility with older experiment formats
COLUMN_RENAMES = {
    'io_ops': 'io_write_ops',
}


def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy columns to current naming convention."""
    renames = {old: new for old, new in COLUMN_RENAMES.items() if old in df.columns and new not in df.columns}
    if renames:
        df = df.rename(columns=renames)
        logger.info(f"  Renamed columns: {renames}")
    return df


def get_run_data(run_id: str, runs_dir: str = RUNS_DIR) -> pd.DataFrame:
    """Load CSV data from a run, harmonizing column names."""
    info = get_run_info(run_id, runs_dir)
    if not info.csv_files:
        raise FileNotFoundError(f"No CSV data in run {run_id}")
    dfs = [harmonize_columns(pd.read_csv(f)) for f in info.csv_files]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    logger.info(f"Loaded {len(df)} samples from run {run_id} ({len(info.csv_files)} file(s))")
    return df


def merge_run_data(run_ids: List[str], runs_dir: str = RUNS_DIR) -> pd.DataFrame:
    """Load and merge CSV data from multiple runs."""
    dfs = []
    for rid in run_ids:
        df = get_run_data(rid, runs_dir)
        df['_source_run'] = rid
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    logger.info(f"Merged {len(merged)} samples from {len(run_ids)} runs")
    return merged


def _format_size(nbytes: int) -> str:
    """Format byte size as human-readable string."""
    if nbytes >= 1_000_000:
        return f"{nbytes / 1_000_000:.1f}M"
    elif nbytes >= 1_000:
        return f"{nbytes / 1_000:.0f}K"
    return f"{nbytes}B"


# ─── CLI Commands ───────────────────────────────────────────

def cmd_list(args):
    """List all runs with data."""
    runs = list_runs(args.runs_dir)
    if not runs:
        print("No runs with data found.")
        return

    # Header
    print(f"{'RUN ID':<20s} {'GRID':<8s} {'SAMPLES':>8s} {'PHASES':<20s} {'SYSID':<6s} {'SIZE':>6s}")
    print("-" * 72)
    for r in runs:
        phases = ','.join(r.phases) if r.phases else '?'
        sysid = 'yes' if r.has_sysid else 'no'
        size = _format_size(r.csv_total_size)
        print(f"{r.run_id:<20s} {r.grid_size:<8s} {r.sample_count:>8d} {phases:<20s} {sysid:<6s} {size:>6s}")


def cmd_inspect(args):
    """Show detailed info for a run."""
    info = get_run_info(args.run_id, args.runs_dir)

    print(f"Run: {info.run_id}")
    print(f"Path: {info.path}")
    print(f"Grid: {info.grid_size} (preset={info.grid_preset})")
    print(f"  batch_size: {info.batch_range}")
    print(f"  poll_interval: {info.poll_range} ms")
    print(f"Phases: {', '.join(info.phases) if info.phases else '?'}")
    print(f"Steps: {info.total_steps} (duration={info.step_duration}s each)")
    print(f"Samples: {info.sample_count}")
    print(f"CSV files: {len(info.csv_files)} ({_format_size(info.csv_total_size)})")
    for f in info.csv_files:
        print(f"  {os.path.basename(f)}")

    if info.sample_count > 0 and info.csv_files:
        # Quick stats
        try:
            df = pd.read_csv(info.csv_files[0])
            n_exhausted = (df['queue_exhausted'].astype(str) == 'True').sum() if 'queue_exhausted' in df.columns else 0
            n_settle = df['phase'].str.contains('settle', na=False).sum() if 'phase' in df.columns else 0
            print(f"  queue_exhausted: {n_exhausted}")
            print(f"  settle samples: {n_settle}")
            print(f"  usable samples: ~{info.sample_count - n_exhausted - n_settle}")
        except Exception:
            pass

    if info.has_sysid:
        print(f"\nSysID output: {os.path.basename(info.sysid_json)}")
        if info.sysid_r_squared:
            print("  R² per state:")
            for var, r2 in info.sysid_r_squared.items():
                print(f"    {var}: {r2:.4f}")
    else:
        print("\nSysID: not yet run")


def cmd_sysid(args):
    """Run system identification on selected run(s)."""
    # Import here to avoid circular imports at module level
    from sysid_analysis import load_data, run_sysid

    run_ids = args.runs if args.runs else [args.run_id]

    if not run_ids:
        print("Error: specify a run_id or --runs <id1> <id2> ...")
        sys.exit(1)

    # Load data
    if len(run_ids) == 1:
        df = get_run_data(run_ids[0], args.runs_dir)
        output_dir = os.path.join(
            args.runs_dir, f'run_{run_ids[0]}', 'results', 'sysid_output'
        )
    else:
        df = merge_run_data(run_ids, args.runs_dir)
        # Save merged output to a shared directory
        output_dir = os.path.join(
            args.runs_dir, 'merged_sysid',
            '_'.join(run_ids)
        )

    if args.output_dir:
        output_dir = args.output_dir

    print(f"\nRunning sysid on {len(run_ids)} run(s): {', '.join(run_ids)}")
    print(f"Total samples: {len(df)}")
    print(f"Output: {output_dir}\n")

    result = run_sysid(
        df, output_dir=output_dir,
        normalize=args.normalize,
        split_ratio=args.split_ratio,
        seed=args.seed,
        filter_exhausted=args.filter_exhausted,
        exclude_settle=args.exclude_settle,
        train_phases=args.train_phases,
    )

    print(f"\nSysID JSON: {result['json_path']}")
    print(f"Report: {result['report_path']}")


def main():
    parser = argparse.ArgumentParser(
        description='Run Selector for LQR and ANN training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python select_run.py list
  python select_run.py inspect 20260303_151634
  python select_run.py sysid 20260303_151634 --normalize
  python select_run.py sysid --runs 20260216_070737 20260303_151634 --normalize
        """
    )
    parser.add_argument('--runs-dir', default=RUNS_DIR,
                        help=f'Runs directory (default: {RUNS_DIR})')

    sub = parser.add_subparsers(dest='command', required=True)

    # list
    sub.add_parser('list', help='List all runs with data')

    # inspect
    p_inspect = sub.add_parser('inspect', help='Show details of a run')
    p_inspect.add_argument('run_id', help='Run ID (e.g., 20260303_151634)')

    # sysid
    p_sysid = sub.add_parser('sysid', help='Run system identification')
    p_sysid.add_argument('run_id', nargs='?', default=None,
                          help='Single run ID')
    p_sysid.add_argument('--runs', nargs='+', default=None,
                          help='Multiple run IDs to merge')
    p_sysid.add_argument('--output-dir', default=None,
                          help='Output directory (default: auto)')
    p_sysid.add_argument('--normalize', action='store_true', default=True,
                          help='Normalize data (default: True)')
    p_sysid.add_argument('--no-normalize', dest='normalize', action='store_false')
    p_sysid.add_argument('--split-ratio', type=float, default=0.8)
    p_sysid.add_argument('--seed', type=int, default=42)
    p_sysid.add_argument('--filter-exhausted', action='store_true', default=True)
    p_sysid.add_argument('--no-filter-exhausted', dest='filter_exhausted',
                          action='store_false')
    p_sysid.add_argument('--exclude-settle', action='store_true', default=True)
    p_sysid.add_argument('--no-exclude-settle', dest='exclude_settle',
                          action='store_false')
    p_sysid.add_argument('--train-phases', nargs='+', default=None)

    args = parser.parse_args()

    if args.command == 'list':
        cmd_list(args)
    elif args.command == 'inspect':
        cmd_inspect(args)
    elif args.command == 'sysid':
        if not args.run_id and not args.runs:
            parser.error("sysid requires a run_id or --runs")
        cmd_sysid(args)


if __name__ == '__main__':
    main()
