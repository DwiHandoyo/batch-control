"""
ANN Training via MLE / Boltzmann Policy from Capacity Benchmark.

Instead of using sysid A, B matrices (unreliable from random excitation),
this script builds a throughput model from the capacity benchmark and uses
it to evaluate cost J(state, u, Q) over a grid of candidate actions.

The training target u* is the Boltzmann expectation:
    P(u | state, Q) ∝ exp(-J(state, u, Q) / T)
    u* = E[u] = Σ_u P(u|state,Q) × u   (expected optimal action)

This is principled MLE: ANN learns the mean of the Boltzmann distribution,
which is the rational action under uncertainty about the exact optimum.

No LQR oracle required. No sysid B matrix used (only normalization scales).

Throughput model (from capacity benchmark, clean data):
    throughput(batch, poll) = throughput_max(batch) × min(1, t_fetch/poll)

Forward simulation (1-step):
    queue_next = max(0, queue - throughput(batch, poll) × dt)
    cpu_next   = cpu_at_benchmark(batch)   [interpolated]

Usage:
    python train_ann_costweighted.py \\
        --csv ../open_loop/results/sysid_data_*.csv \\
        --sysid-json <path>          # normalization scales only
        --capacity-json <path>       # throughput model source
        [--temperature 50]
        [--n-u-grid 15]
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train-ann-mle')

# State + control variable names (must match controllers.py)
STATE_VARS = ['queue_length', 'cpu_util', 'container_mem_pct', 'io_write_ops']
CONTROL_VARS = ['batch_size', 'inv_poll_interval']
N_STATE = len(STATE_VARS)
N_CTRL = len(CONTROL_VARS)

# Target state (must match controllers.py LQRController.x_target)
X_TARGET = np.array([0.0, 5.0, 50.0, 30.0])

# R matrix (must match controllers.py LQRController.R)
# Will be recomputed via Bryson if bounds are available, but used here for
# control-effort penalty in J. Fixed to reasonable base value.
R_BASE = np.diag([0.001, 0.001])

# Q presets (must match controllers.py Q_PRESETS)
Q_PRESETS = {
    'Q1': np.diag([100.0,  1.0, 0.1,  1.0]),
    'Q2': np.diag([ 10.0, 10.0, 0.1, 10.0]),
    'Q4': np.diag([100.0, 10.0, 0.1, 10.0]),
}

# Sample interval (seconds) — matches experiment sample_interval
DT = 1.0


# ──────────────────────────────────────────────────────────────
# Throughput Model (from Capacity Benchmark)
# ──────────────────────────────────────────────────────────────

class ThroughputModel:
    """
    Empirical throughput model from capacity benchmark.

    throughput(batch, poll) = throughput_max(batch) × poll_efficiency(poll)

    poll_efficiency(poll) = min(1, t_fetch / poll)
    Rationale: when poll < t_fetch (Kafka fetch RTT), the poll finishes
    before data is ready → wasted polling, efficiency < 1. When
    poll ≥ t_fetch, each poll finds data → efficiency = 1.

    throughput_max(batch) is interpolated from benchmark data.
    cpu(batch) is interpolated from benchmark data (for cost computation).
    """

    def __init__(self, capacity_json: str):
        with open(capacity_json) as f:
            cap = json.load(f)

        bench = cap['benchmark_results']
        self.t_fetch_ms = cap['measurements']['t_fetch_ms']

        batches = np.array([b['batch_size'] for b in bench], dtype=float)
        throughputs = np.array([b['throughput_msg_per_s'] for b in bench], dtype=float)
        cpus = np.array([b['avg_cpu'] for b in bench], dtype=float)

        # Sort by batch size
        idx = np.argsort(batches)
        batches, throughputs, cpus = batches[idx], throughputs[idx], cpus[idx]

        self.batch_min = float(cap['control_ranges']['batch_min'])
        self.batch_max = float(cap['control_ranges']['batch_max'])
        self.poll_min_ms = float(cap['control_ranges']['poll_min_ms'])
        self.poll_max_ms = float(cap['control_ranges']['poll_max_ms'])

        # Interpolators (linear, extrapolate clipped)
        self._throughput_fn = interp1d(
            batches, throughputs, kind='linear',
            bounds_error=False,
            fill_value=(throughputs[0], throughputs[-1]),
        )
        self._cpu_fn = interp1d(
            batches, cpus, kind='linear',
            bounds_error=False,
            fill_value=(cpus[0], cpus[-1]),
        )

        logger.info(
            f"ThroughputModel loaded: {len(batches)} batch levels, "
            f"t_fetch={self.t_fetch_ms:.1f}ms, "
            f"batch_range=[{self.batch_min:.0f},{self.batch_max:.0f}], "
            f"poll_range=[{self.poll_min_ms:.0f},{self.poll_max_ms:.0f}]ms"
        )

    def throughput(self, batch: float, poll_ms: float) -> float:
        """Effective throughput in msg/s."""
        tmax = float(self._throughput_fn(np.clip(batch, self.batch_min, self.batch_max)))
        # Poll efficiency: min(1, t_fetch / poll)
        eff = min(1.0, self.t_fetch_ms / max(poll_ms, 1.0))
        return tmax * eff

    def cpu(self, batch: float) -> float:
        """Expected CPU utilization (%) for given batch size."""
        return float(self._cpu_fn(np.clip(batch, self.batch_min, self.batch_max)))

    def u_grid(self, n: int = 15) -> np.ndarray:
        """
        2D grid of (batch, inv_poll) candidates, log-spaced within bounds.
        Returns array of shape (n*n, 2) with [batch, inv_poll] pairs.
        """
        batches = np.geomspace(self.batch_min, self.batch_max, n)
        # Convert poll bounds to inv_poll (inv_poll = 1000/poll_ms)
        inv_min = 1000.0 / self.poll_max_ms
        inv_max = 1000.0 / self.poll_min_ms
        inv_polls = np.geomspace(inv_min, inv_max, n)
        bg, ig = np.meshgrid(batches, inv_polls)
        return np.column_stack([bg.ravel(), ig.ravel()])


# ──────────────────────────────────────────────────────────────
# Cost Function J (using throughput model)
# ──────────────────────────────────────────────────────────────

def compute_cost_throughput(
    state: np.ndarray,
    u: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x_target: np.ndarray,
    u_nominal: np.ndarray,
    model: ThroughputModel,
    dt: float = DT,
) -> float:
    """
    1-step cost using throughput model as forward predictor.

    State: [queue, cpu, mem, io] in RAW units.
    u: [batch, inv_poll] in RAW units.

    queue_next = max(0, queue - throughput(batch, poll_ms) × dt)
    cpu_next   = cpu_benchmark(batch)   (steady-state under batch)
    mem_next   = mem (uncontrolled, carry forward)
    io_next    = io  (carry forward, roughly constant)

    Cost = (x_next - x_target)' Q (x_next - x_target)
         + (u - u_nom)' R (u - u_nom)
    One-sided penalty on cpu, mem, io (only overshoot penalized).
    """
    batch = u[0]
    poll_ms = 1000.0 / max(u[1], 0.001)

    tput = model.throughput(batch, poll_ms)
    queue_next = max(0.0, state[0] - tput * dt)
    cpu_next = model.cpu(batch)
    mem_next = state[2]   # uncontrolled
    io_next = state[3]    # uncontrolled

    x_next = np.array([queue_next, cpu_next, mem_next, io_next])

    e = x_next - x_target
    for i in [1, 2, 3]:  # one-sided: cpu, mem, io
        e[i] = max(0.0, e[i])

    du = u - u_nominal
    return float(e @ Q @ e + du @ R @ du)


# ──────────────────────────────────────────────────────────────
# Boltzmann Policy: compute u* = E[u | state, Q]
# ──────────────────────────────────────────────────────────────

def boltzmann_expected_u(
    state: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x_target: np.ndarray,
    u_nominal: np.ndarray,
    model: ThroughputModel,
    u_grid: np.ndarray,
    temperature: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Boltzmann expected action u* = Σ_u P(u|state,Q) × u.

    P(u|state,Q) = softmax(-J(state,u,Q) / T)

    Returns (u_star, costs) where costs is the J value for each u in u_grid.
    """
    costs = np.array([
        compute_cost_throughput(state, u, Q, R, x_target, u_nominal, model)
        for u in u_grid
    ])

    # Numerically stable softmax.
    # Scale temperature by cost range so T is "fraction of cost range"
    # rather than absolute. This makes T=1.0 mean "full spread over cost range"
    # and T=0.01 mean "almost deterministic argmin".
    cost_range = costs.max() - costs.min() + 1e-10
    log_w = -costs / (temperature * cost_range)
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()

    u_star = (w[:, None] * u_grid).sum(axis=0)
    return u_star, costs


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

def load_and_aggregate(csv_path: str, exclude_settle: bool = True,
                       filter_exhausted: bool = True) -> pd.DataFrame:
    """Load open-loop CSV and aggregate per control step (mean state)."""
    df = pd.read_csv(csv_path)
    n0 = len(df)

    if exclude_settle and 'phase' in df.columns:
        df = df[~df['phase'].astype(str).str.contains('settle', na=False)]
        logger.info(f"Excluded settle: {n0} -> {len(df)}")

    if filter_exhausted and 'queue_exhausted' in df.columns:
        df = df[df['queue_exhausted'].astype(str) != 'True']
        logger.info(f"Filtered exhausted: -> {len(df)}")

    df = df.reset_index(drop=True)
    changed = (df[['batch_size', 'poll_interval']] !=
               df[['batch_size', 'poll_interval']].shift(1)).any(axis=1)
    df['step_id'] = changed.cumsum()

    agg = df.groupby('step_id').agg({
        'queue_length': 'mean',
        'cpu_util': 'mean',
        'container_mem_pct': 'mean',
        'io_write_ops': 'mean',
        'batch_size': 'first',
        'poll_interval': 'first',
    }).reset_index(drop=True)

    agg['inv_poll_interval'] = 1000.0 / agg['poll_interval'].clip(lower=1)
    logger.info(f"Aggregated to {len(agg)} steps from {len(df)} samples")
    return agg


def augment_high_queue_states(
    df: pd.DataFrame,
    u_nominal: np.ndarray,
    n_candidates: int = 7,
    high_queue_quantile: float = 0.5,
) -> pd.DataFrame:
    """
    Add synthetic (state, u_placeholder) rows for high-queue coverage.
    u values are placeholders — actual targets computed via Boltzmann.
    Uses log-spaced u scales around u_nominal.
    """
    threshold = df['queue_length'].quantile(high_queue_quantile)
    high = df[df['queue_length'] >= threshold].reset_index(drop=True)
    logger.info(f"Augmenting {len(high)} high-queue states (threshold={threshold:.0f})")

    scales = np.geomspace(0.1, 5.0, n_candidates)
    rows = []
    for i in range(len(high)):
        x = high.iloc[i][STATE_VARS].values.astype(float)
        for s in scales:
            u_cand = u_nominal * s
            rows.append({
                'queue_length': float(x[0]),
                'cpu_util': float(x[1]),
                'container_mem_pct': float(x[2]),
                'io_write_ops': float(x[3]),
                'batch_size': float(u_cand[0]),
                'inv_poll_interval': float(u_cand[1]),
                'poll_interval': 1000.0 / max(u_cand[1], 0.001),
            })
    aug = pd.DataFrame(rows)
    logger.info(f"Augmented {len(aug)} rows ({len(high)} states × {n_candidates})")
    return aug


def build_q_grid(n_random: int = 20, seed: int = 42) -> List[Tuple[str, np.ndarray]]:
    """3 presets + N random Q matrices."""
    rng = np.random.RandomState(seed)
    grid = list(Q_PRESETS.items())
    bounds = [(10.0, 1000.0), (0.5, 50.0), (0.01, 1.0), (0.5, 50.0)]
    for i in range(n_random):
        diag = np.array([
            np.exp(rng.uniform(np.log(lo), np.log(hi)))
            for lo, hi in bounds
        ])
        grid.append((f'Qrand{i:02d}', np.diag(diag)))
    logger.info(f"Q grid: {len(grid)} ({len(Q_PRESETS)} presets + {n_random} random)")
    return grid


# ──────────────────────────────────────────────────────────────
# Build training dataset
# ──────────────────────────────────────────────────────────────

def normalize_with_sysid(df: pd.DataFrame, sysid_json: str):
    """Normalize state and control vectors using sysid scales."""
    with open(sysid_json) as f:
        sysid = json.load(f)
    norm = sysid.get('normalization', {})

    states_raw = df[STATE_VARS].values.astype(np.float64)
    controls_raw = df[CONTROL_VARS].values.astype(np.float64)

    states_norm = np.zeros_like(states_raw)
    for i, k in enumerate(STATE_VARS):
        if k in norm and norm[k]['std'] > 0:
            states_norm[:, i] = (states_raw[:, i] - norm[k]['mean']) / norm[k]['std']
        else:
            states_norm[:, i] = states_raw[:, i]

    controls_norm = np.zeros_like(controls_raw)
    for i, k in enumerate(CONTROL_VARS):
        if k in norm and norm[k]['std'] > 0:
            controls_norm[:, i] = (controls_raw[:, i] - norm[k]['mean']) / norm[k]['std']
        else:
            controls_norm[:, i] = controls_raw[:, i]

    return states_norm, controls_norm, norm


def normalize_u(u_raw: np.ndarray, norm: Dict) -> np.ndarray:
    out = np.zeros(N_CTRL)
    for i, k in enumerate(CONTROL_VARS):
        if k in norm and norm[k]['std'] > 0:
            out[i] = (u_raw[i] - norm[k]['mean']) / norm[k]['std']
        else:
            out[i] = u_raw[i]
    return out


def build_training_set(
    df: pd.DataFrame,
    q_grid: List,
    model: ThroughputModel,
    u_nominal: np.ndarray,
    temperature: float,
    n_u_grid: int,
    sysid_json: str,
) -> Dict:
    """
    Build (X, y) training arrays via Boltzmann MLE.

    For each (state, Q):
      - Evaluate J over u-grid using throughput model
      - Compute Boltzmann weights = softmax(-J / T)
      - Compute u* = Σ w × u  (Boltzmann expected action)
      - Normalize u* for ANN output

    X: [Q_diag_log10 (4), state_norm (4)]  → 8-dim
    y: u*_norm (2)
    """
    states_norm, _, norm = normalize_with_sysid(df, sysid_json)
    states_raw = df[STATE_VARS].values.astype(np.float64)

    u_nom_norm = normalize_u(u_nominal, norm)

    # Build Bryson R for cost computation (raw space, so use raw bounds)
    u_min_raw = np.array([model.batch_min, 1000.0 / model.poll_max_ms])
    u_max_raw = np.array([model.batch_max, 1000.0 / model.poll_min_ms])
    delta_u_max = np.maximum(u_max_raw - u_nominal, u_nominal - u_min_raw)
    R_bryson = np.diag(1.0 / (delta_u_max ** 2))
    logger.info(f"Bryson R (raw space): {np.diag(R_bryson)}")

    # u-grid in raw space for Boltzmann evaluation
    u_grid_raw = model.u_grid(n_u_grid)   # shape (n*n, 2): [batch, inv_poll]

    X_list, y_list = [], []
    cost_min_list = []

    n_states = len(df)
    n_q = len(q_grid)
    logger.info(f"Computing Boltzmann u* for {n_states} states × {n_q} Q = {n_states*n_q} pairs "
                f"(u-grid: {len(u_grid_raw)} points)...")

    for qi, (q_label, Q) in enumerate(q_grid):
        q_diag_log = np.log10(np.diag(Q) + 1e-3)
        u_stars_raw = []
        costs_min = []

        for i in range(n_states):
            x_raw = states_raw[i]
            u_star_raw, costs = boltzmann_expected_u(
                x_raw, Q, R_bryson, X_TARGET, u_nominal, model,
                u_grid_raw, temperature,
            )
            u_stars_raw.append(u_star_raw)
            costs_min.append(costs.min())

        if (qi + 1) % 5 == 0 or qi == 0:
            logger.info(f"  Q [{qi+1}/{n_q}] {q_label}: "
                        f"cost_min=[{min(costs_min):.2f}, {np.median(costs_min):.2f}]")

        for i in range(n_states):
            u_star_norm = normalize_u(u_stars_raw[i], norm)
            X_list.append(np.concatenate([q_diag_log, states_norm[i]]))
            y_list.append(u_star_norm)
            cost_min_list.append(costs_min[i])

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)

    logger.info(f"Dataset: {len(X)} pairs")
    logger.info(f"Cost_min stats: min={min(cost_min_list):.2f}, "
                f"median={np.median(cost_min_list):.2f}, max={max(cost_min_list):.2f}")
    logger.info(f"u_star (normalized) stats:\n"
                f"  batch_norm: mean={y[:,0].mean():.3f}, std={y[:,0].std():.3f}\n"
                f"  poll_norm:  mean={y[:,1].mean():.3f}, std={y[:,1].std():.3f}")

    return {'X': X, 'y': y}


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def train_model(X: np.ndarray, y: np.ndarray,
                hidden_dim: int = 128, epochs: int = 300, lr: float = 0.001,
                batch_size: int = 256, val_ratio: float = 0.1, seed: int = 42) -> Dict:
    """Train 8→hidden→hidden→2 network with MSE loss (no weighting needed — y is already u*)."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        logger.error("PyTorch not installed.")
        sys.exit(1)

    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(X)
    n_val = max(1, int(n * val_ratio))
    idx = np.random.permutation(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train = torch.FloatTensor(X[train_idx])
    y_train = torch.FloatTensor(y[train_idx])
    X_val = torch.FloatTensor(X[val_idx])
    y_val = torch.FloatTensor(y[val_idx])

    ds = TensorDataset(X_train, y_train)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = nn.Sequential(
        nn.Linear(X.shape[1], hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, y.shape[1]),
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info(f"Training ANN: {X.shape[1]}→{hidden_dim}→{hidden_dim}→{y.shape[1]}")

    best_val, best_state = float('inf'), None
    history = {'train': [], 'val': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = sum(
            (lambda pred, loss: loss.item())(
                pred := model(xb),
                (criterion(pred, yb).backward() or optimizer.step() or
                 optimizer.zero_grad() or criterion(pred, yb))
            )
            for xb, yb in loader
        ) if False else 0.0  # placeholder — use proper loop below
        # Proper loop:
        epoch_loss = 0.0
        nb = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1
        train_loss = epoch_loss / nb

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 50 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1:>4d}: train={train_loss:.6f} val={val_loss:.6f}")

    model.load_state_dict(best_state)
    model.eval()
    layers = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            layers.append({'weight': param.detach().numpy().tolist()})
        elif 'bias' in name:
            layers[-1]['bias'] = param.detach().numpy().tolist()

    logger.info(f"Training done. Best val={best_val:.6f}")
    return {
        'layers': layers,
        'training_info': {
            'epochs': epochs, 'lr': lr, 'hidden_dim': hidden_dim,
            'best_val_loss': float(best_val),
            'final_train_loss': float(history['train'][-1]),
        },
        'history': {'train': [float(x) for x in history['train']],
                    'val': [float(x) for x in history['val']]},
    }


def export_model(model_data: Dict, normalization: Dict, capacity_json: str,
                 q_grid: List, output_path: str) -> str:
    export = {
        'model_type': 'ann_universal_qaware',
        'layers': model_data['layers'],
        'normalization': normalization,
        'state_vars': STATE_VARS,
        'control_vars': CONTROL_VARS,
        'q_input_transform': 'log10(diag + 1e-3)',
        'input_dim': N_STATE * 2,
        'architecture': f"{N_STATE*2}-{model_data['training_info']['hidden_dim']}-"
                        f"{model_data['training_info']['hidden_dim']}-{N_CTRL}",
        'training_method': 'boltzmann_mle_capacity_benchmark',
        'training_info': model_data['training_info'],
        'q_presets_used': {label: np.diag(Q).tolist() for label, Q in q_grid[:len(Q_PRESETS)]},
        'n_random_q': len(q_grid) - len(Q_PRESETS),
        'capacity_json': capacity_json,
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_path, 'w') as f:
        json.dump(export, f, indent=2)
    logger.info(f"Exported model to {output_path}")
    return output_path


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train Q-aware ANN via Boltzmann MLE + capacity benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--csv', required=True, help='Open-loop CSV path')
    parser.add_argument('--sysid-json', required=True,
                        help='Sysid JSON (normalization scales only)')
    parser.add_argument('--capacity-json', required=True,
                        help='Capacity benchmark JSON (throughput model source)')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--temperature', type=float, default=50.0,
                        help='Boltzmann temperature T (smaller=more selective)')
    parser.add_argument('--n-u-grid', type=int, default=15,
                        help='Grid points per control dimension (total = n^2)')
    parser.add_argument('--n-random-q', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--u-nominal', nargs=2, type=float, default=[250.0, 5.0])
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--no-augment', dest='augment', action='store_false')
    parser.add_argument('--n-candidates', type=int, default=7)
    parser.add_argument('--high-queue-quantile', type=float, default=0.5)
    parser.add_argument('--verify', action='store_true',
                        help='Print cost landscape verification before training')

    args = parser.parse_args()
    u_nominal = np.array(args.u_nominal)

    # 1. Build throughput model
    logger.info(f"Loading capacity benchmark: {args.capacity_json}")
    model = ThroughputModel(args.capacity_json)

    # 2. Load + aggregate open-loop data (state distribution source)
    logger.info(f"Loading open-loop data: {args.csv}")
    df = load_and_aggregate(args.csv)

    if args.augment:
        aug = augment_high_queue_states(df, u_nominal, args.n_candidates,
                                        args.high_queue_quantile)
        df = pd.concat([df, aug], ignore_index=True)
        logger.info(f"Combined dataset: {len(df)} states")

    # 3. Q grid
    q_grid = build_q_grid(args.n_random_q, args.seed)

    # 4. Optional: verify cost landscape
    if args.verify:
        print("\n=== COST LANDSCAPE VERIFICATION ===")
        Q1 = Q_PRESETS['Q1']
        u_grid = model.u_grid(10)
        from scipy.interpolate import interp1d as _  # already imported
        u_min_raw = np.array([model.batch_min, 1000.0 / model.poll_max_ms])
        u_max_raw = np.array([model.batch_max, 1000.0 / model.poll_min_ms])
        delta_u = np.maximum(u_max_raw - u_nominal, u_nominal - u_min_raw)
        R = np.diag(1.0 / (delta_u ** 2))
        for q_raw in [500, 5000, 20000, 100000]:
            x = np.array([float(q_raw), 5.0, 50.0, 30.0])
            u_star, costs = boltzmann_expected_u(x, Q1, R, X_TARGET, u_nominal,
                                                 model, u_grid, args.temperature)
            print(f"  queue={q_raw:>7.0f}: u*=[batch={u_star[0]:.0f}, "
                  f"inv_poll={u_star[1]:.2f}→poll={1000/max(u_star[1],0.001):.0f}ms], "
                  f"cost_min={costs.min():.1f}")
        print()

    # 5. Build training set
    data = build_training_set(df, q_grid, model, u_nominal, args.temperature,
                              args.n_u_grid, args.sysid_json)

    # 6. Get normalization for export
    _, _, normalization = normalize_with_sysid(df, args.sysid_json)

    # 7. Train
    model_data = train_model(
        data['X'], data['y'],
        hidden_dim=args.hidden_dim, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, seed=args.seed,
    )

    # 8. Export
    output_dir = args.output_dir or os.path.dirname(args.sysid_json)
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'ann_universal_mle_{ts}.json')
    export_model(model_data, normalization, args.capacity_json, q_grid, output_path)

    print(f"\nTraining complete!")
    print(f"  Best val loss: {model_data['training_info']['best_val_loss']:.6f}")
    print(f"  Model saved:   {output_path}")


if __name__ == '__main__':
    main()
